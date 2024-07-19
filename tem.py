"""
From: https://github.com/thuiar/Self-MM
Paper: Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis
"""
# self supervised multimodal multi-task learning network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..subNets import BertTextEncoder

__all__ = ['AFN_AG_MSA']
class StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, aug_mean, aug_var, K = 1):
        N, L, C = x.size()
        x = x.permute(0, 2, 1)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)


            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1) / K
            if x.is_cuda:
                alpha = alpha.cuda()
            mean = (1 - alpha) * mean + alpha * aug_mean
            var = (1 - alpha) * var + alpha * aug_var

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, L)
        x = x.permute(0, 2, 1)
        return x
class DecoderLayer(nn.Module):
    r"""DecoderLayer is mainly made up of the proposed cross-modal relation attention (CMRA).

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model=32, nhead=2, dim_feedforward=64, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer.
        """
        #memory = torch.cat([memory, tgt], dim=0)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class AFN_AG_MSA(nn.Module):
    def __init__(self, args):
        super(AFN_AG_MSA, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.hidden_dim = 32
        self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)
        self.text_linear = nn.Linear(args.text_out, self.hidden_dim)
        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_in = audio_in
        self.video_in = video_in
        self.audio_self_att = nn.MultiheadAttention(embed_dim=self.audio_in, num_heads=1, batch_first=True)
        self.video_self_att = nn.MultiheadAttention(embed_dim=self.video_in, num_heads=1, batch_first=True)
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, self.hidden_dim, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, self.hidden_dim, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)
        self.audio_cross_att = DecoderLayer()
        self.video_cross_att = DecoderLayer()

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(self.hidden_dim*3, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(self.hidden_dim, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(self.hidden_dim, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(self.hidden_dim, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

        self.SAG_Transformer_1 = SAG_Transformer(hidden_dim=self.hidden_dim)
        self.SAG_Transformer_2 = SAG_Transformer(hidden_dim=self.hidden_dim)
        self.audio_adaIN = StyleRandomization()
        self.video_adaIN = StyleRandomization()

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        audio_dim = audio.size(2)
        visual_dim = video.size(2)
        visual_mean = video.permute(0, 2, 1).mean(-1, keepdim=True)
        visual_mean = visual_mean.mean(1).unsqueeze(1).repeat((1, audio_dim, 1))
        visual_var = video.permute(0, 2, 1).var(-1, keepdim=True)
        visual_var = visual_var.mean(1).unsqueeze(1).repeat((1, audio_dim, 1))
        acoustic_mean = audio.permute(0, 2, 1).mean(-1, keepdim=True)
        acoustic_mean = acoustic_mean.mean(1).unsqueeze(1).repeat((1, visual_dim, 1))

        acoustic_var = audio.permute(0, 2, 1).var(-1, keepdim=True)
        acoustic_var = acoustic_var.mean(1).unsqueeze(1).repeat((1, visual_dim, 1))
        video = self.video_adaIN(video, acoustic_mean, acoustic_var, K=5)
        audio = self.audio_adaIN(audio, visual_mean, visual_var, K=5)

        audio = self.audio_self_att(audio, audio, audio)[0]
        video = self.video_self_att(video, video, video)[0]

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze(1).int().detach().cpu()
        #text = self.text_model(text)[:,0,:]
        text = self.text_model(text)
        text = self.text_linear(text)
        # print("text:", text.size())
        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)
        audio, text_audio_hidden = self.SAG_Transformer_1(text, audio)
        video, text_video_hidden = self.SAG_Transformer_2(text, video)
        # print("audio:", audio.size())
        # print("video:", video.size())

        text = torch.cat([text_audio_hidden, text_video_hidden], dim=1)
        text = text.mean(dim=1)
        audio = audio.mean(dim=1)
        video = video.mean(dim=1)
        # print("new_text:", text.size())
        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion,
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        #_, final_states = self.rnn(packed_sequence)
        final_states, _, = self.rnn(packed_sequence)
        final_states,_ = pad_packed_sequence(final_states, batch_first=True)
        #h = self.dropout(final_states[0].squeeze(0))
        h = self.dropout(final_states)
        y_1 = self.linear_1(h)
        return y_1


class SAG_Transformer(nn.Module):
    def __init__(self, hidden_dim):
        super(SAG_Transformer, self).__init__()
        self.tgt_self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.tgt_self_attention_1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.text_self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.text_self_attention_1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.tgt_gru = nn.GRU(input_size=hidden_dim, hidden_size=int(hidden_dim/2), bidirectional=True, batch_first=True)
        self.text_gru = nn.GRU(input_size=hidden_dim, hidden_size=int(hidden_dim/2), bidirectional=True, batch_first=True)
        self.tgt_retrain_gate = nn.Linear(hidden_dim, 1)
        self.tgt_compound_gate = nn.Linear(hidden_dim, 1)
        self.text_retrain_gate = nn.Linear(hidden_dim, 1)
        self.text_compound_gate = nn.Linear(hidden_dim,1)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, text, target):


        text_hidden, _ = self.text_gru(text)
        text_retrain_gate = self.sigmoid(self.text_retrain_gate(text_hidden))
        text_compound_gate = self.sigmoid(self.text_compound_gate(text_hidden))
        text_attention = self.text_self_attention(text_hidden, text_hidden, text_hidden)[0]
        new_text_hidden = self.norm(text_retrain_gate * text_hidden + text_compound_gate * text_attention)
        new_text_hidden = self.text_self_attention_1(new_text_hidden, new_text_hidden, new_text_hidden)[0]

        tgt_hidden, _ = self.tgt_gru(target)
        tgt_retrain_gate = self.sigmoid(self.tgt_retrain_gate(tgt_hidden))
        tgt_compound_gate = self.sigmoid(self.tgt_compound_gate(tgt_hidden))
        target_attention = self.tgt_self_attention(tgt_hidden, tgt_hidden, tgt_hidden)[0]
        target_attention = self.tgt_self_attention_1(target_attention, target_attention, target_attention)[0]

        tgt_cross_attention = self.cross_attention(target_attention, text_attention, text_attention)[0]
        new_tgt_hidden = self.norm(tgt_retrain_gate * tgt_hidden + tgt_compound_gate * tgt_cross_attention)



        return new_tgt_hidden, new_text_hidden





