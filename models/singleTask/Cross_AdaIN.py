import torch
import torch.nn as nn

class StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, K = 1):
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
            mean = (1 - alpha) * mean + alpha * mean[idx_swap]
            var = (1 - alpha) * var + alpha * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, L)
        x = x.permute(0, 2, 1)
        return x


class Cross_StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, text, audio, video, K = 1):
        N, L, C_t = text.size()
        N, L, C_a = audio.size()
        N, L, C_v = video.size()
        text = text.permute(0, 2, 1)
        audio = audio.permute(0, 2, 1)
        video = video.permute(0, 2, 1)
        if self.training:

            text_mean = text.mean(-1, keepdim=True)
            text_var = text.var(-1, keepdim=True)
            text = (text - text_mean) / (text_var + self.eps).sqrt()

            audio_mean = audio.mean(-1, keepdim=True)
            audio_var = audio.var(-1, keepdim=True)
            audio = (audio - audio_mean) / (audio_var + self.eps).sqrt()

            video_mean = video.mean(-1, keepdim=True)
            video_var = video.var(-1, keepdim=True)
            video = (video - video_mean) / (video_var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            text_alpha = torch.rand(N, 1, 1) / K
            audio_alpha = torch.randn(N, 1, 1) / K
            video_alpha = torch.randn(N, 1, 1) / K
            if text.is_cuda:
                text_alpha = text_alpha.cuda()
                audio_alpha = audio_alpha.cuda()
                video_alpha = video_alpha.cuda()

            text_mean_new = (1 - text_alpha) * text_mean + text_alpha/2 * audio_mean + text_alpha/2 * video_mean
            text_var_new = (1 - text_alpha) * text_var + text_alpha/2 * audio_var + text_alpha/2 * video_var

            audio_mean_new = (1 - audio_alpha) * audio_mean + audio_alpha/2 * text_mean + audio_alpha/2 * video_mean
            audio_var_new = (1 - audio_alpha) * audio_var + audio_alpha/2 * text_var + audio_alpha/2 * video_var

            video_mean_new = (1 - video_alpha) * video_mean + video_alpha/2 * text_mean + video_alpha/2 * audio_mean
            video_var_new = (1 - video_alpha) * video_var + video_alpha/2 * video_var + video_alpha/2 * audio_var

            text = text * (text_var_new + self.eps).sqrt() + text_mean_new
            text = text.view(N, C_t, L)
            audio = audio * (audio_var_new + self.eps).sqrt() + audio_mean_new
            audio = audio.view(N, C_a, L)
            video = video * (video_var_new + self.eps).sqrt() + video_mean_new
            video = video.view(N, C_v, L)


        text = text.permute(0, 2, 1)
        audio = audio.permute(0, 2, 1)
        video = video.permute(0, 2, 1)
        return text, audio, video

x = torch.randn(32, 20, 512)
AdaIN = StyleRandomization()
x = AdaIN(x)
print(x.size())

text = torch.randn(32, 20, 1024)
audio = torch.randn(32, 20, 512)
video = torch.randn(32, 20, 512)
model = Cross_StyleRandomization()
_,_,_ = model(text, audio, video)