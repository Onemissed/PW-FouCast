import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DCNv3 import DCNv3_pytorch
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from timm.models.layers import DropPath

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ResDWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = Residual(nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim))

    def forward(self, x):
        return self.dwconv(x)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.fn(x))
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        if not hidden_dim:
            hidden_dim = dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)
    
class ConvFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.):
        super().__init__()
        if not hidden_dim:
            hidden_dim = dim
        self.conv_1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.dwconv = ResDWConv(hidden_dim)
        self.conv_2 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )
        self.droprateout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_1(x)
        x = self.dwconv(x)
        x = self.conv_2(x)
        x = self.droprateout(x)
        return x.permute(0, 2, 3, 1)

class TransBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.attention_block = Residual(PostNorm(dim, NeighborhoodAttention(dim=dim,
                                                                            kernel_size=7,
                                                                            dilation=None,
                                                                            num_heads=heads,
                                                                            qkv_bias=True,
                                                                            qk_scale=None,
                                                                            attn_drop=0.0,
                                                                            proj_drop=0.0
                                                                            )))
        
        self.mlp_block = Residual(PostNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class GlobalBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.transblock1 = TransBlock(dim=dim, heads=heads, mlp_dim=mlp_dim)
        self.transblock2 = TransBlock(dim=dim, heads=heads, mlp_dim=mlp_dim)
    
    def forward(self, x):
        x = self.transblock1(x)
        x = self.transblock2(x)
        return x

class DCNBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.attention_block = Residual(PostNorm(dim, DCNv3_pytorch(dim)))
        self.mlp_block = Residual(PostNorm(dim, ConvFeedForward(dim=dim,
                                                                hidden_dim=mlp_dim)))
    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x
    
class LocalBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.convblock_1 = DCNBlock(dim=dim, mlp_dim=mlp_dim)
        self.convblock_2 = DCNBlock(dim=dim, mlp_dim=mlp_dim)
        
    def forward(self, x):
        x = self.convblock_1(x)
        x = self.convblock_2(x)
        return x
    

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        attn = self.dwconv(x)
        attn = self.conv_spatial(attn)
        attn = self.pwconv(attn)
        return x * attn
    

class ConvNeXt_block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class GLFJM(nn.Module):
    def __init__(self, dim, kernel_size=1, stride=1, padding=0):
        super(GLFJM, self).__init__()
        self.conv_fusion = SpatialAttention(dim * 2)
        self.att_g = nn.Conv2d(dim, dim, kernel_size, stride, padding)
        self.att_l = nn.Conv2d(dim, dim, kernel_size, stride, padding)
        self.conv_g = ConvNeXt_block(dim)
        self.conv_l = ConvNeXt_block(dim)

    def forward(self, g, l):
        cat = torch.cat((g, l), dim=1)
        fusion = self.conv_fusion(cat) + cat
        g_att, l_att = torch.chunk(fusion, 2, dim=1)
        g_att = torch.sigmoid(self.att_g(g_att)) * self.conv_g(g)
        l_att = torch.sigmoid(self.att_l(l_att)) * self.conv_l(l)
        return g_att + l_att
    

class SelfAttention(nn.Module):
    def __init__(self, dim: int=192, ratio_kq: int=8, ratio_v: int=8):
        super(SelfAttention, self).__init__()
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.conv_q = nn.Conv2d(dim, dim//ratio_kq, 1, 1, 0, bias=False)
        self.conv_k = nn.Conv2d(dim, dim//ratio_kq, 1, 1, 0, bias=False)
        self.conv_v = nn.Conv2d(dim, dim//ratio_v, 1, 1, 0, bias=False)
        self.conv_out = nn.Conv2d(dim//ratio_v, dim, 1, 1, 0, bias=False)
    
    def einsum(self, q, k, v):
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        beta = torch.einsum("bchw, bcL->bLhw", q, k)
        beta = torch.softmax(beta, dim=1)
        out = torch.einsum("bLhw, bcL->bchw", beta, v)
        return out

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        att = self.einsum(q, k, v)
        att = F.sigmoid(self.conv_out(att))
        return att * x + x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = torch.reshape(x, (b, c, new_h, self.downscaling_factor, new_w, self.downscaling_factor))
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = torch.reshape(x, (b, c * (self.downscaling_factor ** 2), new_h, new_w)).permute(0, 2, 3, 1)
        x = x.to(torch.float32)
        x = self.linear(x)
        return x


class PatchExpanding(nn.Module):
    def __init__(self, in_channels, out_channels, upscaling_factor):
        super().__init__()
        self.upscaling_factor = upscaling_factor
        self.linear = nn.Linear(in_channels // (upscaling_factor ** 2), out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h * self.upscaling_factor, w * self.upscaling_factor
        new_c = int(c // (self.upscaling_factor ** 2))
        x = torch.reshape(x, (b, new_c, self.upscaling_factor, self.upscaling_factor, h, w))
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = torch.reshape(x, (b, new_c, new_h, new_w)).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class SkipAttention(nn.Module):
    def __init__(self, dim, downscaling_factor):
        super().__init__()

        self.dim = dim
        down_up = []
        for i, factor in enumerate(downscaling_factor): 
            if factor > 0:
                down_up.append(PatchMerging(int(dim / factor), dim, factor))
            elif factor < 0:
                down_up.append(PatchExpanding(int(dim * (-factor)), dim, -factor))
            else:
                down_up.append(nn.Identity())
        self.down_up = nn.ModuleList(down_up)
        self.norm = nn.LayerNorm(dim * 4)
        self.att = SelfAttention(dim * 4)
        self.smoothconv = nn.Conv2d(dim * 4, dim, kernel_size=1)

    def forward(self, xs, anchor):
        ans = []
        for i, x in enumerate(xs):
            x = self.down_up[i](x)
            if x.shape[-1] / self.dim == 1:
                x = x.permute(0, 3, 1, 2)
            ans.append(x)
        for i in range(3):
            ans[i] = ans[i] + ans[i+1]
        ans = torch.cat(ans, dim=1)
        ans = ans.permute(0, 2, 3, 1)
        ans = self.norm(ans)
        ans = ans.permute(0, 3, 1, 2)
        ans = self.att(ans)
        return self.smoothconv(ans) + anchor
    
class AMSFM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.att1 = SkipAttention(dim=hidden_dim * 1, downscaling_factor=(0, -2, -4, -8))
        self.att2 = SkipAttention(dim=hidden_dim * 2, downscaling_factor=(2, 0, -2, -4))
        self.att3 = SkipAttention(dim=hidden_dim * 4, downscaling_factor=(4, 2, 0, -2))
        self.att4 = SkipAttention(dim=hidden_dim * 8, downscaling_factor=(8, 4, 2, 0))
    
    def forward(self, x1, x2, x3, x4):
        out4 = self.att4([x1, x2, x3, x4], x4)
        out3 = self.att3([x1, x2, x3, x4], x3)
        out2 = self.att2([x1, x2, x3, x4], x2)
        out1 = self.att1([x1, x2, x3, x4], x1)
        return out1, out2, out3, out4


class SkipAttention_pangu(nn.Module):
    def __init__(self, dim, downscaling_factor):
        super().__init__()

        self.dim = dim
        down_up = []
        for i, factor in enumerate(downscaling_factor):
            if factor > 0:
                down_up.append(PatchMerging(int(dim / factor), dim, factor))
            elif factor < 0:
                down_up.append(PatchExpanding(int(dim * (-factor)), dim, -factor))
            else:
                down_up.append(nn.Identity())
        self.down_up = nn.ModuleList(down_up)
        self.norm = nn.LayerNorm(dim * 3)
        self.att = SelfAttention(dim * 3)
        self.smoothconv = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, xs, anchor):
        ans = []
        for i, x in enumerate(xs):
            x = self.down_up[i](x)
            if x.shape[-1] / self.dim == 1:
                x = x.permute(0, 3, 1, 2)
            ans.append(x)
        for i in range(1):
            ans[i] = ans[i] + ans[i+1]
        ans = torch.cat(ans, dim=1)
        ans = ans.permute(0, 2, 3, 1)
        ans = self.norm(ans)
        ans = ans.permute(0, 3, 1, 2)
        ans = self.att(ans)
        return self.smoothconv(ans) + anchor


class AMSFM_pangu(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att1 = SkipAttention_pangu(dim=hidden_dim * 2, downscaling_factor=(0, -2, -4))
        self.att2 = SkipAttention_pangu(dim=hidden_dim * 4, downscaling_factor=(2, 0, -2))
        self.att3 = SkipAttention_pangu(dim=hidden_dim * 8, downscaling_factor=(4, 2, 0))

    def forward(self, x1, x2, x3):
        out3 = self.att3([x1, x2, x3], x3)
        out2 = self.att2([x1, x2, x3], x2)
        out1 = self.att1([x1, x2, x3], x1)
        return out1, out2, out3


class Stage_Down(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads):
        super().__init__()
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                ResDWConv(dim=hidden_dimension),
                GlobalBlock(dim=hidden_dimension, heads=num_heads, mlp_dim=hidden_dimension * 4),
                LocalBlock(dim=hidden_dimension, mlp_dim=hidden_dimension * 4),
                GLFJM(hidden_dimension)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for dwconv, global_block, local_block, glfjm in self.layers:
            x = x.permute(0, 3, 1, 2)
            x = dwconv(x)
            x = x.permute(0, 2, 3, 1)
            local_x = local_block(x)
            local_x = local_x.permute(0, 3, 1, 2)
            global_x = global_block(x)
            global_x = global_x.permute(0, 3, 1, 2)
            out = glfjm(global_x, local_x)
        return out
    

class Stage_Up(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, upscaling_factor, num_heads):
        super().__init__()
        self.patch_partition = PatchExpanding(in_channels=in_channels, out_channels=hidden_dimension,
                                              upscaling_factor=upscaling_factor)

        self.in_channel = in_channels
        self.hidden_dimension = hidden_dimension
        
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                ResDWConv(dim=hidden_dimension * 2),
                GlobalBlock(dim=hidden_dimension * 2, heads=num_heads, mlp_dim=hidden_dimension * 4),
                LocalBlock(dim=hidden_dimension * 2, mlp_dim=hidden_dimension * 4),
                GLFJM(hidden_dimension * 2)
            ]))

    def forward(self, x, skip_x):
        x = self.patch_partition(x)
        skip_x = skip_x.permute(0, 2, 3, 1)
        x = torch.cat((x, skip_x), dim=-1)
        for dwconv, global_block, local_block, glfjm in self.layers:
            x = x.permute(0, 3, 1, 2)
            x = dwconv(x)
            x = x.permute(0, 2, 3, 1)
            local_x = local_block(x)
            local_x = local_x.permute(0, 3, 1, 2)
            global_x = global_block(x)
            global_x = global_x.permute(0, 3, 1, 2)
            out = glfjm(global_x, local_x)
        return out


class Stage_Out(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, upscaling_factor, num_heads):
        super().__init__()
        self.patch_partition = PatchExpanding(in_channels=in_channels, out_channels=hidden_dimension,
                                              upscaling_factor=upscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                ResDWConv(dim=hidden_dimension),
                GlobalBlock(dim=hidden_dimension, heads=num_heads, mlp_dim=hidden_dimension * 4),
                LocalBlock(dim=hidden_dimension, mlp_dim=hidden_dimension * 4),
                GLFJM(hidden_dimension)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for dwconv, global_block, local_block, glfjm in self.layers:
            x = x.permute(0, 3, 1, 2)
            x = dwconv(x)
            x = x.permute(0, 2, 3, 1)
            local_x = local_block(x)
            local_x = local_x.permute(0, 3, 1, 2)
            global_x = global_block(x)
            global_x = global_x.permute(0, 3, 1, 2)
            out = glfjm(global_x, local_x)
        return out


class STJointEncoder(nn.Module):
    def __init__(self, input_channel, hidden_dim, downscaling_factors, layers, heads):
        super(STJointEncoder, self).__init__()
        self.stage1 = Stage_Down(in_channels=input_channel, hidden_dimension=hidden_dim, layers=layers[0],
                                 downscaling_factor=downscaling_factors[0], num_heads=heads[0])

        self.stage2 = Stage_Down(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                 downscaling_factor=downscaling_factors[1], num_heads=heads[1])

        self.stage3 = Stage_Down(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                 downscaling_factor=downscaling_factors[2], num_heads=heads[2])

        self.stage4 = Stage_Down(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                 downscaling_factor=downscaling_factors[3], num_heads=heads[3])
        
    def forward(self, x):
        x1 = self.stage1(x)     # (4, 96, 32, 32)
        x2 = self.stage2(x1)    # (4, 192, 16, 16)
        x3 = self.stage3(x2)    # (4, 384, 8, 8)
        x4 = self.stage4(x3)    # (4, 768, 4, 4)
        return x1, x2, x3, x4


class STJointEncoder_pangu(nn.Module):
    def __init__(self, input_channel, hidden_dim, downscaling_factors, layers, heads):
        super(STJointEncoder_pangu, self).__init__()
        self.stage1 = Stage_Down(in_channels=input_channel, hidden_dimension=hidden_dim * 2, layers=layers[0],
                                 downscaling_factor=downscaling_factors[0], num_heads=heads[0])

        self.stage2 = Stage_Down(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[1],
                                 downscaling_factor=downscaling_factors[1], num_heads=heads[1])

        self.stage3 = Stage_Down(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[2],
                                 downscaling_factor=downscaling_factors[2], num_heads=heads[2])

    def forward(self, x):
        x1 = self.stage1(x)  # (4, 96, 32, 32)
        x2 = self.stage2(x1)  # (4, 192, 16, 16)
        x3 = self.stage3(x2)  # (4, 192, 16, 16)
        return x1, x2, x3


class STJointDecoder(nn.Module):
    def __init__(self, hidden_dim, output_channel, downscaling_factors, layers, heads):
        super(STJointDecoder, self).__init__()
        self.stage1 = Stage_Up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 4,
                               layers=layers[3], upscaling_factor=downscaling_factors[3], num_heads=heads[3])

        self.stage2 = Stage_Up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 2,
                               layers=layers[2], upscaling_factor=downscaling_factors[2], num_heads=heads[2])

        self.stage3 = Stage_Up(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 1,
                               layers=layers[1], upscaling_factor=downscaling_factors[1], num_heads=heads[1])

        self.stage4 = Stage_Out(in_channels=hidden_dim * 2, hidden_dimension=output_channel,
                                layers=layers[0], upscaling_factor=downscaling_factors[0], num_heads=heads[0])
    
    def forward(self, x, x3, x2, x1):
        x = self.stage1(x, x3)    # (4, 768, 8, 8)
        x = self.stage2(x, x2)    # (4, 384, 16, 16)
        x = self.stage3(x, x1)    # (4, 192, 32, 32)
        x = self.stage4(x)        # (4, 12, 128, 128)
        return x
    

class STJointNet(nn.Module):
    def __init__(self, input_channel, hidden_dim, output_channel, downscaling_factors, layers, heads):
        super(STJointNet, self).__init__()
        self.encoder = STJointEncoder(input_channel=input_channel, hidden_dim=hidden_dim, layers=layers,
                                      downscaling_factors=downscaling_factors, heads=heads)

        self.decoder = STJointDecoder(hidden_dim=hidden_dim, output_channel=output_channel, layers=layers, 
                                      downscaling_factors=downscaling_factors, heads=heads)
        
        self.amsfm = AMSFM(hidden_dim=hidden_dim)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        b, t, c, h, w = x.shape
        x = torch.reshape(x, [b, t * c, h, w])
        x = x.view(b, t * c, h, w)
        
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.amsfm(x1, x2, x3, x4)
        x = self.decoder(x4, x3, x2, x1)

        x = torch.reshape(x, [b, -1, c, h, w])
        return x

class MMSTJointDecoder(nn.Module):
    def __init__(self, hidden_dim, output_channel, downscaling_factors, layers, heads):
        super(MMSTJointDecoder, self).__init__()
        self.echo_stage1 = Stage_Up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 4,
                               layers=layers[3], upscaling_factor=downscaling_factors[3], num_heads=heads[3])

        self.echo_stage2 = Stage_Up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 2,
                               layers=layers[2], upscaling_factor=downscaling_factors[2], num_heads=heads[2])

        self.echo_stage3 = Stage_Up(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 1,
                               layers=layers[1], upscaling_factor=downscaling_factors[1], num_heads=heads[1])

        self.echo_stage4 = Stage_Out(in_channels=hidden_dim * 2, hidden_dimension=output_channel,
                                layers=layers[0], upscaling_factor=downscaling_factors[0], num_heads=heads[0])
        
        self.sate_stage1 = Stage_Up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 4,
                               layers=layers[3], upscaling_factor=downscaling_factors[3], num_heads=heads[3])

        self.sate_stage2 = Stage_Up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 2,
                               layers=layers[2], upscaling_factor=downscaling_factors[2], num_heads=heads[2])

    def forward(self, x, x3, x2, x1, y, y3, y2):
        x = x + y
        x = self.echo_stage1(x, x3)    # (4, 768, 8, 8)
        y = self.sate_stage1(y, y3)    # (4, 768, 8, 8)
        x = x + y

        x = self.echo_stage2(x, x2)    # (4, 384, 16, 16)
        y = self.sate_stage2(y, y2)    # (4, 384, 16, 16)
        x = x + y

        x = self.echo_stage3(x, x1)    # (4, 192, 32, 32)
        # y = self.sate_stage3(y, y1)    # (4, 192, 32, 32)

        x = self.echo_stage4(x)        # (4, 12, 128, 128)
        # y = self.sate_stage4(y)        # (4, 12, 128, 128)

        return x
    

class CM_STJointNet(nn.Module):
    def __init__(self, input_channel, hidden_dim, output_channel, downscaling_factors, layers, heads):
        super(CM_STJointNet, self).__init__()
        self.x_encoder = STJointEncoder(input_channel=input_channel, hidden_dim=hidden_dim, layers=layers,
                                            downscaling_factors=downscaling_factors, heads=heads)
        self.y_encoder = STJointEncoder_pangu(input_channel=400, hidden_dim=hidden_dim, layers=[2, 2, 2],
                                              downscaling_factors=[1, 2, 2], heads=[2, 4, 8])
        
        self.x_skip = AMSFM(hidden_dim=hidden_dim)
        self.y_skip = AMSFM_pangu(hidden_dim=hidden_dim)
        
        self.mm_decoder = MMSTJointDecoder(hidden_dim=hidden_dim, output_channel=output_channel, layers=layers, 
                                            downscaling_factors=downscaling_factors, heads=heads)

    def forward(self, x, y):
        b, t, c, h, w = x.shape
        x = torch.reshape(x, [b, t * c, h, w])

        _, t_, c_, h_, w_ = y.shape
        y = torch.reshape(y, [b, t_ * c_, h_, w_])

        x1, x2, x3, x4 = self.x_encoder(x)
        y1, y2, y3 = self.y_encoder(y)
        x1, x2, x3, x4 = self.x_skip(x1, x2, x3, x4)
        y1, y2, y3 = self.y_skip(y1, y2, y3)
        
        x = self.mm_decoder(x4, x3, x2, x1, y3, y2, y1)
        x = torch.reshape(x, [b, -1, c, h, w])

        return x