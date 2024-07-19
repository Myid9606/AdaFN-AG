"""
paper: Memory Fusion Network for Multi-View Sequential Learning
From: https://github.com/pliang279/MFN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MFN']

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
class MFN(nn.Module):
    def __init__(self, args):
        super(MFN, self).__init__()
        self.d_l,self.d_a,self.d_v = args.feature_dims
        self.dh_l,self.dh_a,self.dh_v = args.hidden_dims
        total_h_dim = self.dh_l+self.dh_a+self.dh_v
        self.mem_dim = args.memsize
        window_dim = args.windowsize
        output_dim = args.num_classes if args.train_mode == "classification" else 1
        attInShape = total_h_dim*window_dim
        gammaInShape = attInShape+self.mem_dim
        final_out = total_h_dim+self.mem_dim
        h_att1 = args.NN1Config["shapes"]
        h_att2 = args.NN2Config["shapes"]
        h_gamma1 = args.gamma1Config["shapes"]
        h_gamma2 = args.gamma2Config["shapes"]
        h_out = args.outConfig["shapes"]
        att1_dropout = args.NN1Config["drop"]
        att2_dropout = args.NN2Config["drop"]
        gamma1_dropout = args.gamma1Config["drop"]
        gamma2_dropout = args.gamma2Config["drop"]
        out_dropout = args.outConfig["drop"]

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)
        self.audio_adaIN = StyleRandomization()
        self.video_adaIN = StyleRandomization()
        
    def forward(self, text_x, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, sequence_len, audio_in)
            video_x: tensor of shape (batch_size, sequence_len, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''

        text_x = text_x.permute(1,0,2)
        audio_x = audio_x.permute(1,0,2)
        video_x = video_x.permute(1,0,2)

        # *********************************Ada IN*******************************
        audio_dim = audio_x.size(2)
        visual_dim = video_x.size(2)
        visual_mean = video_x.permute(0, 2, 1).mean(-1, keepdim=True)
        visual_mean = visual_mean.mean(1).unsqueeze(1).repeat((1, audio_dim, 1))
        visual_var = video_x.permute(0, 2, 1).var(-1, keepdim=True)
        visual_var = visual_var.mean(1).unsqueeze(1).repeat((1, audio_dim, 1))
        acoustic_mean = audio_x.permute(0, 2, 1).mean(-1, keepdim=True)
        acoustic_mean = acoustic_mean.mean(1).unsqueeze(1).repeat((1, visual_dim, 1))

        acoustic_var = audio_x.permute(0, 2, 1).var(-1, keepdim=True)
        acoustic_var = acoustic_var.mean(1).unsqueeze(1).repeat((1, visual_dim, 1))
        
        # video_x = self.video_adaIN(video_x, acoustic_mean, acoustic_var, K=3)    
        audio_x = self.audio_adaIN(audio_x, visual_mean, visual_var, K=3)
        # *********************************Ada IN*******************************
        # x is t x n x d
        n = text_x.size()[1]
        t = text_x.size()[0]
        self.h_l = torch.zeros(n, self.dh_l).to(text_x.device)
        self.h_a = torch.zeros(n, self.dh_a).to(text_x.device)
        self.h_v = torch.zeros(n, self.dh_v).to(text_x.device)
        self.c_l = torch.zeros(n, self.dh_l).to(text_x.device)
        self.c_a = torch.zeros(n, self.dh_a).to(text_x.device)
        self.c_v = torch.zeros(n, self.dh_v).to(text_x.device)
        self.mem = torch.zeros(n, self.mem_dim).to(text_x.device)
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(t):
            # prev time step
            prev_c_l = self.c_l
            prev_c_a = self.c_a
            prev_c_v = self.c_v
            # curr time step
            new_h_l, new_c_l = self.lstm_l(text_x[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(audio_x[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(video_x[i], (self.h_v, self.c_v))
            # concatenate
            prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
            new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
            cStar = torch.cat([prev_cs,new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention*cStar
            cHat = torch.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended,self.mem], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            all_mems.append(self.mem)
            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)

        # last hidden layer last_hs is n x h
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        res = {
            'M': output,
            'L': last_hs
        }
        return res