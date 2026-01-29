import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.nowcastnet_module.utils import warp, make_grid
from module.nowcastnet_module.generation.generative_network import Generative_Encoder, Generative_Decoder
from module.nowcastnet_module.evolution.evolution_network import Evolution_Network
from module.nowcastnet_module.generation.noise_projector import Noise_Projector

# -----------------------------------------------------------------------------
# Paper: Y. Zhang, M. Long, K. Chen, L. Xing, R. Jin, M. I. Jordan, and J. Wang (2023)
#        Skilful nowcasting of extreme precipitation with nowcastnet
#        https://www.nature.com/articles/s41586-023-06184-4
# -----------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, args, ngf):
        super(Net, self).__init__()
        self.args = args
        self.pred_length = self.args.total_length - self.args.input_length

        self.evo_net = Evolution_Network(self.args.input_length, self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.args.total_length, base_c=ngf)
        self.gen_dec = Generative_Decoder(self.args)
        self.proj = Noise_Projector(ngf, args)

        sample_tensor = torch.zeros(1, 1, self.args.img_width, self.args.img_width)
        self.grid = make_grid(sample_tensor)
        self.ngf = ngf

    def forward(self, all_frames):
        all_frames = all_frames[:, :, :, :, :1]

        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        input_frames = frames[:, :self.args.input_length]
        input_frames = input_frames.reshape(batch, self.args.input_length, height, width)

        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)
        series = []
        last_frames = all_frames[:, (self.args.input_length - 1):self.args.input_length, :, :, 0]
        grid = self.grid.repeat(batch, 1, 1, 1)

        for i in range(self.pred_length):
            last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)

        # Normalization for evolution network output
        # evo_result = evo_result / 255

        evo_feature = self.gen_enc(torch.cat([input_frames, evo_result], dim=1))

        noise = torch.randn(batch, self.ngf, height // 32, width // 32).cuda()
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_result)

        return gen_result.unsqueeze(-1)