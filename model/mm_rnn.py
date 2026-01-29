import torch
from torch import nn

class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_channel, output_channel, h_w, kernel_size, stride, padding):
        super().__init__()
        self._state_height, self._state_width = h_w
        self._conv_x2h = nn.Conv2d(in_channels=input_channel, out_channels=output_channel * 4,
                                   kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_h2h = nn.Conv2d(in_channels=input_channel, out_channels=output_channel * 4,
                                   kernel_size=kernel_size, stride=stride, padding=padding)
        self._input_channel = input_channel
        self._output_channel = output_channel

    def forward(self, x, hiddens):
        if hiddens is None:
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        else:
            h, c = hiddens
        x2h = self._conv_x2h(x)
        h2h = self._conv_h2h(h)
        i, f, g, o = torch.chunk((x2h + h2h), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        next_c = f * c + i * g

        o = torch.sigmoid(o)
        next_h = o * torch.tanh(next_c)

        ouput = next_h
        next_hiddens = [next_h, next_c]

        return ouput, next_hiddens


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5  # scale factor for dot-product

        # Projections for query (from "target" sequence), key/value (from "source" sequence)
        self.q_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        self.k_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        self.v_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _shape_for_heads(self, x: torch.Tensor):
        x = x.view(B, L, self.num_heads, self.head_dim)     # (B, L, heads, head_dim)
        x = x.transpose(1, 2)                               # (B, heads, L, head_dim)
        return x

    def forward(self, query, key):
        # Basic checks
        B, C, H, W = query.shape
        query = query.permute(0, 2, 3, 1).reshape(B, -1, C)
        key = key.permute(0, 2, 3, 1).reshape(B, -1, C)

        B, L_q, _ = query.shape
        _, L_k, _ = key.shape
        value = key

        # Project
        q = self.q_proj(query)   # (B, L_q, E)
        k = self.k_proj(key)     # (B, L_k, E)
        v = self.v_proj(value)   # (B, L_k, E)

        # Split into heads: (B, heads, L, head_dim)
        q = self._shape_for_heads(q)
        k = self._shape_for_heads(k)
        v = self._shape_for_heads(v)

        # scale Q
        q = q * self.scaling

        # Compute attention scores: (B, heads, L_q, L_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        # Softmax -> attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, heads, L_q, L_k)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)        # (B, heads, L_q, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        attn_output = attn_output + query + key
        attn_output = attn_output.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return attn_output


class MM_ConvLSTM(nn.Module):
    def __init__(self, input_channel, output_channel, h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = 6
        H, W = h_w
        lstm = [ConvLSTM_Cell(input_channel, output_channel, [H, W], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H // 2, W // 2], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H // 4, W // 4], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H // 4, W // 4], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H // 2, W // 2], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)

        lstm_pangu = [ConvLSTM_Cell(input_channel, output_channel, [H, W], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H // 2, W // 2], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H // 4, W // 4], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H // 4, W // 4], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H // 2, W // 2], kernel_size, stride, padding),
                ConvLSTM_Cell(input_channel, output_channel, [H, W], kernel_size, stride, padding)]
        self.lstm_pangu = nn.ModuleList(lstm_pangu)
        self.MFM1 = MultiHeadCrossAttention(embed_dim=output_channel, num_heads=4)
        self.MFM2 = MultiHeadCrossAttention(embed_dim=output_channel, num_heads=4)
        self.MFM3 = MultiHeadCrossAttention(embed_dim=output_channel, num_heads=4)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.downs_pangu = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])
        self.ups_pangu = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

    def forward(self, x, x_pangu, layer_hiddens, layer_hiddens_pangu, embed, embed_pangu, fc):
        x = embed(x)
        x_pangu = embed_pangu(x_pangu)
        next_layer_hiddens = []
        next_layer_hiddens_pangu = []
        out = []
        out_pangu = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None

            if layer_hiddens_pangu is not None:
                hiddens_pangu = layer_hiddens_pangu[l]
            else:
                hiddens_pangu = None

            x_pangu, next_hiddens_pangu = self.lstm_pangu[l](x_pangu, hiddens_pangu)
            out_pangu.append(x_pangu)

            if l == 3:
                x = self.MFM1(query=x, key=x_pangu)
            elif l == 4:
                x = self.MFM2(query=x, key=x_pangu)
            elif l == 5:
                x = self.MFM3(query=x, key=x_pangu)

            x, next_hiddens = self.lstm[l](x, hiddens)
            out.append(x)
            if l == 0:
                x = self.downs[0](x)
                x_pangu = self.downs_pangu[0](x_pangu)
            elif l == 1:
                x = self.downs[1](x)
                x_pangu = self.downs_pangu[1](x_pangu)
            elif l == 3:
                x = self.ups[0](x) + out[1]
                x_pangu = self.ups_pangu[0](x_pangu) + out_pangu[1]
            elif l == 4:
                x = self.ups[1](x) + out[0]
                x_pangu = self.ups_pangu[1](x_pangu) + out_pangu[0]
            next_layer_hiddens.append(next_hiddens)
            next_layer_hiddens_pangu.append(next_hiddens_pangu)
        x = fc(x)

        return x, next_layer_hiddens, next_layer_hiddens_pangu


class Model(nn.Module):
    def __init__(self, args, in_shape, num_hidden):
        super().__init__()
        T, C, H, W = in_shape
        self.rnns = MM_ConvLSTM(input_channel=num_hidden[0], output_channel=num_hidden[0], h_w=[H, W],
                                kernel_size=args.filter_size, stride=1, padding=2)
        self.embed = nn.Conv2d(in_channels=args.patch_size * args.patch_size, out_channels=num_hidden[0], kernel_size=1, stride=1, padding=0, dilation=1)
        self.embed_pangu = nn.Conv2d(in_channels=20, out_channels=num_hidden[0], kernel_size=1, stride=1, padding=0, dilation=1)
        self.fc = nn.Conv2d(in_channels=num_hidden[0], out_channels=args.patch_size * args.patch_size, kernel_size=1, stride=1, padding=0, dilation=1)
        self.args = args

    def forward(self, x, x_pangu, mask_true):
        # T, B, C, H, W
        x = x.permute(1, 0, 4, 2, 3)
        x_pangu = x_pangu.permute(1, 0, 2, 3, 4)
        mask_true = mask_true.permute(1, 0, 4, 2, 3)

        outputs = []
        layer_hiddens = None
        layer_hiddens_pangu = None
        output = None
        for t in range(self.args.input_length - 1, x.shape[0] - 1):
            if t < self.args.input_length:
                input = x[t]
            else:
                input = mask_true[t - self.args.input_length] * x[t] + (1 - mask_true[t - self.args.input_length]) * output

            input_pangu = x_pangu[t - self.args.input_length + 1]
            output, layer_hiddens, layer_hiddens_pangu = self.rnns(input, input_pangu, layer_hiddens, layer_hiddens_pangu, self.embed, self.embed_pangu, self.fc)
            outputs.append(output)
        outputs = torch.stack(outputs)  # s b c h w
        outputs = outputs.permute(1, 0, 3, 4, 2)

        return outputs