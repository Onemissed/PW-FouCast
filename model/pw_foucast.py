import torch
import math
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# Basic Convolution Block
class BasicConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True,
                 enc=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        # Use PixelShuffle for upsampling
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
            self.norm = nn.GroupNorm(2, out_channels)
        elif enc is True:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation)
            self.conv3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, dilation=dilation))
            self.norm1 = nn.GroupNorm(2, out_channels)
            self.norm2 = nn.GroupNorm(2, out_channels)
            self.norm3 = nn.GroupNorm(2, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)
            self.norm = nn.GroupNorm(2, out_channels)

        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)
        self.upsampling = upsampling
        self.enc = enc

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.enc:
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x3 = self.conv3(x)
            if self.act_norm:
                x1 = self.act(self.norm1(x1))
                x2 = self.act(self.norm2(x2))
                x3 = self.act(self.norm3(x3))
            y = torch.cat([x1, x2, x3], dim=1)
        else:
            y = self.conv(x)
            if self.act_norm:
                y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True,
                 enc=False):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace, enc=enc)

    def forward(self, x):
        y = self.conv(x)
        return y


class PW_FouCast(nn.Module):
    def __init__(self, args, in_shape, hid_S=16, N_S=4, N_T=4, mlp_ratio=8.,
                 drop=0.0, drop_path=0.0, spatio_kernel_enc=3, spatio_kernel_dec=3):
        super(PW_FouCast, self).__init__()
        T, C, H, W = in_shape
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.enc_pangu = Encoder_pangu(20, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(T, args.output_length, hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.hid = MidMetaNet(T, T * hid_S, args.output_length * hid_S, N_T,
                              input_resolution=(H, W),
                              mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.out_T = args.output_length
        self.memory = Fourier_memory(C, hid_S, N_S, spatio_kernel_enc, T, T * hid_S, args.output_length * hid_S)

    def forward(self, x_raw, pangu, gt):
        B, T, C, H, W = x_raw.shape
        x = x_raw.contiguous().view(B * T, C, H, W)

        pangu = pangu.reshape(-1, 20, 32, 32)
        embed_pangu = self.enc_pangu(pangu)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        z_pangu = embed_pangu.view(B, 20, C_, H_, W_)

        if gt is not None:
            match_phase = self.memory(x_raw, gt)
        else:
            match_phase = self.memory(x_raw, None)

        hid = self.hid(z, z_pangu, match_phase)
        hid = hid.reshape(B * self.out_T, C_, H_, W_)

        Y = self.dec(hid, skip)

        Y = Y.reshape(B, self.out_T, C, H, W)

        return Y

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace, enc=True),
            *[ConvSC(3 * C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace, enc=True) for s in samplings[1:]]
        )
        self.fuse1 = nn.Conv2d(3 * C_hid, C_hid, 1)
        self.fuse2 = nn.Conv2d(3 * C_hid, C_hid, 1)

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        latent = self.fuse1(latent)
        enc1 = self.fuse2(enc1)
        return latent, enc1

# Pangu-Weather forecasts encoder
class Encoder_pangu(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        super(Encoder_pangu, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=False,
                     act_inplace=act_inplace, enc=True),
            *[ConvSC(3 * C_hid, C_hid, spatio_kernel, downsampling=False,
                     act_inplace=act_inplace, enc=True)]
        )
        self.fuse1 = nn.Conv2d(3 * C_hid, C_hid, 1)

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        latent = self.fuse1(latent)
        return latent


class Decoder(nn.Module):
    def __init__(self, in_T, out_T, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        self.in_T = in_T
        self.out_T = out_T

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)

        B, C, H, W = enc1.shape
        enc1 = enc1.reshape(B // self.in_T, self.in_T, C, H, W)
        # Copy the last frame of shallow features from the encoder T' times to serve as residuals
        enc1 = enc1[:, -1:].repeat(1, self.out_T, 1, 1, 1)
        enc1 = enc1.reshape(-1, C, H, W)

        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MetaBlock(nn.Module):
    def __init__(self, T, in_channels, out_channels, input_resolution=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = AFNOBlock(T, in_channels, out_channels, spatio_kernel=3, embed_dim=out_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path=drop_path, layer_i=layer_i)

    def forward(self, x, x_pangu, z_fft):
        z = self.block(x, x_pangu, z_fft)
        return z


class MidMetaNet(nn.Module):
    def __init__(self, T, channel_in, channel_hid, N2,
                 input_resolution=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        enc_layers = [MetaBlock(T,
            channel_in, channel_hid, input_resolution,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2):
            enc_layers.append(MetaBlock(T,
                channel_hid, channel_hid, input_resolution,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))

        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x, x_pangu, match_phase):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z_pangu = x_pangu.reshape(B, 20 * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z, z_pangu, match_phase)

        y = z.reshape(B, -1, C, H, W)
        return y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 重点：ANFO中使用的模块
class AFNO_SubBlock(nn.Module):
    def __init__(self, T, C_in, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_fno=False, use_blocks=False, layer_i=0):
        super().__init__()
        self.norm1 = norm_layer(C_in)
        self.filter = AFNO2D(T=T, C_in=C_in, dim=dim, hidden_size=dim, num_blocks=1, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, layer_i=layer_i)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.double_skip = True

        self.input_dim = C_in
        self.hidden_dim = dim
        self.proj = nn.Linear(C_in, dim)

    # ANFO模块的前向传播过程
    def forward(self, x, x_pangu, match_phase):
        residual = x

        if self.input_dim != self.hidden_dim:
            residual = self.proj(residual)

        x = self.norm1(x)
        x = self.filter(x, x_pangu, match_phase)

        if self.double_skip:
            x = self.drop_path(x) + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNO2D(nn.Module):
    def __init__(self, T, C_in, dim, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, layer_i=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, C_in, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        if layer_i == 0:
            self.w3 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
            self.b3 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
            # Phase-Fusion Coefficient
            self.alpha = nn.Parameter(torch.full((1, 1, 1, hidden_size), 0.5), requires_grad=True)

        self.T = T
        self.layer_i = layer_i

        if layer_i != 0:
            self.hf_weight = nn.Parameter(torch.full((1, 1, 1, 1, hidden_size, 2), 1e-2), requires_grad=True)

    def wrap_phase(self, delta):
        return torch.remainder(delta + math.pi, 2 * math.pi) - math.pi

    def forward(self, x, x_pangu, match_phase, spatial_size=None):
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        if spatial_size == None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.reshape(B, H, W, C)
        x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, C)

        o1_real = (
            torch.einsum('...bi,bio->...bo', x.real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[1]) +
            self.b1[0]
        )
        o1_imag = (
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x.real, self.w1[1]) +
            self.b1[1]
        )

        """Pangu-Weather-guided Frequency Modulation (in the first layer)"""
        if self.layer_i == 0:
            x = torch.stack([o1_real, o1_imag], dim=-1)
            x = torch.view_as_complex(x)
            x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)

            x_pangu = torch.fft.fft2(x_pangu, dim=(1, 2), norm="ortho")

            x_pangu = x_pangu.reshape(B, x_pangu.shape[1], x_pangu.shape[2], self.num_blocks, self.hidden_size)
            temp_real = (
                    torch.einsum('...bi,bio->...bo', x_pangu.real, self.w3[0]) -
                    torch.einsum('...bi,bio->...bo', x_pangu.imag, self.w3[1]) +
                    self.b3[0]
            )
            temp_imag = (
                    torch.einsum('...bi,bio->...bo', x_pangu.imag, self.w3[0]) +
                    torch.einsum('...bi,bio->...bo', x_pangu.real, self.w3[1]) +
                    self.b3[1]
            )
            x_pangu = torch.stack([temp_real, temp_imag], dim=-1)
            x_pangu = torch.view_as_complex(x_pangu)
            x_pangu = x_pangu.reshape(B, x_pangu.shape[1], x_pangu.shape[2], self.hidden_size)

            eps = 1e-8
            # Calculate similarity (inner product)
            cross_attention = x_pangu * x.conj()
            mag_pangu = x_pangu.abs()
            mag_ori = x.abs()

            # Perform phase fusion below
            phi_x = torch.angle(x)
            phi_pangu = torch.angle(x_pangu)
            cosA = torch.cos(phi_x)
            sinA = torch.sin(phi_x)
            cosB = torch.cos(phi_pangu)
            sinB = torch.sin(phi_pangu)

            sum_cos = self.alpha * cosA + (1 - self.alpha) * cosB
            sum_sin = self.alpha * sinA + (1 - self.alpha) * sinB
            norm = torch.sqrt(sum_cos ** 2 + sum_sin ** 2 + eps)
            # Normalize the results to make the magnitude of the fused phasor equal to 1 (unit phasor)
            out_cos = sum_cos / norm
            out_sin = sum_sin / norm

            # Normalize similarity
            norm_similarity = cross_attention / (mag_pangu * mag_ori + eps)
            # Take the real part of the inner product as the raw similarity
            norm_similarity = norm_similarity.real
            soft_similarity = F.softmax(norm_similarity, dim=3)

            mag_ori = mag_ori * soft_similarity

            # Reconstructing Fourier coefficients
            real1 = mag_ori * out_cos
            imag1 = mag_ori * out_sin
            x = torch.complex(real1, imag1)

            x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.hidden_size)
            o1_real = x.real
            o1_imag = x.imag

        """Frequency Memory-Phase Alignement (in the second layer)"""
        if self.layer_i == 1:
            x = torch.stack([o1_real, o1_imag], dim=-1)
            x = torch.view_as_complex(x)
            x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)

            u_m = torch.view_as_complex(match_phase)
            # Feature phasor and magnitude
            mag_x = torch.abs(x)  # real >=0
            u_x = x / (mag_x + 1e-8)  # unit phasor of x (complex)

            # Complex similarity (unit phasor correlation)
            sim_complex = u_m * torch.conj(u_x)
            sim_real = sim_complex.real  # cos(delta) in [-1,1]

            w = 1.0 - 0.5 * (1.0 + sim_real)

            phi_m = torch.angle(u_m)
            phi_x = torch.angle(u_x)
            delta = self.wrap_phase(phi_m - phi_x)

            rot = torch.cos(w * delta) + 1j * torch.sin(w * delta)
            x = x * rot

            x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.hidden_size)
            o1_real = x.real
            o1_imag = x.imag

        o1_real = F.relu(o1_real)
        o1_imag = F.relu(o1_imag)
        
        original = torch.stack([o1_real, o1_imag], dim=-1)

        o2_real = (
            torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) +
            self.b2[0]
        )
        o2_imag = (
            torch.einsum('...bi,bio->...bo', o1_imag, self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real, self.w2[1]) +
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)

        """Inverted Frequency Attention"""
        if self.layer_i != 0:
            # Calculate the removed high-frequency components
            hf = original - x
            x = x + hf * self.hf_weight

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.ifft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, -1)
        x = x.real
        x = x.type(dtype)

        return x


class Fourier_memory(nn.Module):
    def __init__(self, C, hid_S, N_S, spatio_kernel_enc, T, C_in, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=True)
        self.norm1 = norm_layer(dim)
        self.memblock = FFTmem_Block(T=T, C_in=C_in, dim=dim, mlp_ratio=8, drop=0., drop_path=0.1, norm_layer=norm_layer)

        self.norm2 = norm_layer(dim)
        self.phase_memory = nn.Parameter(0.5 * torch.randn(240, dim, 2))

    def _to_phasor_from_complex(self, X_complex):
        mag = torch.abs(X_complex)
        # Normalization
        real = X_complex.real / (mag + 1e-8)
        imag = X_complex.imag / (mag + 1e-8)
        ph = torch.stack([real, imag], dim=-1)  # [..., dim, 2]
        return ph, mag

    def forward(self, x, gt):
        B, T, C, H, W = x.shape

        if gt is not None:
            gt = gt.reshape(-1, C, H, W)
            embed_gt, _ = self.enc(gt)
            _, C_, H_, W_ = embed_gt.shape
            z_gt = embed_gt.view(B, -1, H_, W_)
            z_gt = z_gt.flatten(2).transpose(1, 2)
            z_gt = z_gt.reshape(B, H_, W_, -1)
            z_gt = torch.fft.fft2(z_gt, dim=(1, 2), norm="ortho")
            phase_src = z_gt
        else:
            x = x.contiguous().view(B * T, C, H, W)
            embed, _ = self.enc(x)
            _, C_, H_, W_ = embed.shape
            z = embed.view(B, T, C_, H_, W_)
            x = z.reshape(B, -1, H_, W_)
            x = x.flatten(2).transpose(1, 2)
            x = self.memblock(x)
            x = x.reshape(B, H_, W_, -1)
            x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")
            phase_src = x

        ph_src, mag_src = self._to_phasor_from_complex(phase_src)  # [B, H_, W_, dim, 2]

        mem = self.phase_memory  # [M, dim, 2]
        # normalize memory phasors to unit length per vector for stable similarity
        mem_norm = mem / (torch.linalg.norm(mem, dim=(1, 2), keepdim=True) + 1e-8)  # [M,dim,2]

        # compute similarity = sum_over_dim( ph_src * mem ) where product = real*real + imag*imag
        # ph_src [..., dim, 2], mem [M, dim, 2] -> we want [..., M]
        # einsum succinctly:
        # sim_raw[b,h,w,m] = sum_d ( ph_src[b,h,w,d,0]*mem[m,d,0] + ph_src[...,d,1]*mem[m,d,1] )
        sim_raw = torch.einsum('bhwdc,mdc->bhwm', ph_src, mem_norm)  # shape [B,H_,W_,M]

        soft_sim = F.softmax(sim_raw, dim=-1)  # weights over memory slots

        # compute matched phasor via weighted sum of mem_norm (broadcast)
        # result: [B,H,W,dim,2] = sum_m soft_sim[...,m] * mem_norm[m,dim,2]
        match_phase = torch.einsum('bhwm,mdc->bhwdc', soft_sim, mem_norm)  # [B,H,W,dim,2]

        return match_phase

class FFTmem_Block(nn.Module):
    def __init__(self, T, C_in, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(C_in)
        self.filter = AFNO2D_memory(T=T, C_in=C_in, dim=dim, hidden_size=dim, num_blocks=1, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.input_dim = C_in
        self.hidden_dim = dim
        self.proj = nn.Linear(C_in, dim)

    def forward(self, x):
        residual = x
        if self.input_dim != self.hidden_dim:
            residual = self.proj(residual)
        x = self.norm1(x)
        x = self.filter(x)
        x = self.drop_path(x) + residual
        residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNO2D_memory(nn.Module):
    def __init__(self, T, C_in, dim, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, C_in, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.T = T

    def forward(self, x, spatial_size=None):
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        if spatial_size == None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.reshape(B, H, W, C)
        x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, C)

        o1_real = F.relu(
            torch.einsum('...bi,bio->...bo', x.real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[1]) + \
            self.b1[0]
        )
        o1_imag = F.relu(
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x.real, self.w1[1]) + \
            self.b1[1]
        )
        o2_real = (
            torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) +
            self.b2[0]
        )
        o2_imag = (
            torch.einsum('...bi,bio->...bo', o1_imag, self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real, self.w2[1]) +
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)

        x = torch.fft.ifft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, -1)
        x = x.real
        x = x.type(dtype)

        return x

class AFNOBlock(nn.Module):
    def __init__(self, T, C_in, C_hid, spatio_kernel, embed_dim, mlp_ratio, drop_rate, drop_path, norm_layer=nn.LayerNorm, use_fno=False, use_blocks=False, act_inplace=True, layer_i=0):
        super(AFNOBlock, self).__init__()

        self.forward_feature = AFNO_SubBlock(T=T, C_in=C_in, dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=drop_path, norm_layer=norm_layer, use_fno=use_fno, use_blocks=use_blocks, layer_i=layer_i)

    def forward(self, x, x_pangu, match_phase):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x_pangu = x_pangu.permute(0, 2, 3, 1)
        x = self.forward_feature(x, x_pangu, match_phase)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x