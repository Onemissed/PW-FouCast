import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Paper: Y.-a. Geng, Q. Li, T. Lin, L. Jiang, L. Xu, D. Zheng, W. Yao, W. Lyu, and Y. Zhang (2019)
#        Lightnet: A dual spatiotemporal encoder network model for lightning prediction
#        https://dl.acm.org/doi/abs/10.1145/3292500.3330717
# -----------------------------------------------------------------------------

class Encoder_wrf_model(nn.Module):
    def __init__(self, tra_frames, channels, args):
        super(Encoder_wrf_model, self).__init__()
        self.args = args
        self.tra_frames = tra_frames
        self.wrf_encoder_conv2d = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.wrf_encoder_convLSTM2D = ConvLSTM2D(64, 32, kernel_size=5, img_rowcol=(args.img_width // 2) // 2)

    def forward(self, wrf):
        # wrf : [frames, batch_size, channels, x, y]
        batch_size = wrf.shape[1]
        wrf_conv = [None] * self.tra_frames
        for i in range(self.tra_frames):
            wrf_conv[i] = self.wrf_encoder_conv2d(wrf[i])
        wrf_h = torch.zeros([batch_size, 32, wrf_conv[0].shape[2], wrf_conv[0].shape[3]], dtype=torch.float32).to(wrf.device)
        wrf_c = torch.zeros([batch_size, 32, wrf_conv[0].shape[2], wrf_conv[0].shape[3]], dtype=torch.float32).to(wrf.device)

        for i in range(self.tra_frames):
            wrf_h, wrf_c = self.wrf_encoder_convLSTM2D(wrf_conv[i], wrf_h, wrf_c)

        return wrf_h, wrf_c

class Encoder_obs_model(nn.Module):
    def __init__(self, tra_frames, channels, args):
        super(Encoder_obs_model, self).__init__()
        self.args = args
        self.tra_frames = tra_frames
        self.obs_encoder_conv2d = nn.Sequential(
                nn.Conv2d(channels, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.obs_encoder_convLSTM2D = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=(args.img_width // 2) // 2)

    def forward(self, obs):
        # obs : [frames, batch_size, channels, x, y]
        batch_size = obs.shape[1]
        obs_conv = [None] * self.tra_frames
        for i in range(self.tra_frames):
            obs_conv[i] = self.obs_encoder_conv2d(obs[i])
        obs_h = torch.zeros([batch_size, 8, obs_conv[0].shape[2], obs_conv[0].shape[3]], dtype=torch.float32).to(obs.device)
        obs_c = torch.zeros([batch_size, 8, obs_conv[0].shape[2], obs_conv[0].shape[3]], dtype=torch.float32).to(obs.device)

        for i in range(self.tra_frames):
            obs_h, obs_c = self.obs_encoder_convLSTM2D(obs_conv[i], obs_h, obs_c)
        return obs_h, obs_c


class Fusion_model(nn.Module):
    def __init__(self, filters_list_input, filters_list_output, args):
        super(Fusion_model, self).__init__()
        self.args = args
        self.filters_list_output = filters_list_output
        fusion_conv_h = [None] * len(self.filters_list_output)
        fusion_conv_c = [None] * len(self.filters_list_output)
        # fusion_conv_encoder = [None] * len(self.filters_list)
        for i in range(len(self.filters_list_output)):
            fusion_conv_h[i] = nn.Sequential(
                nn.Conv2d(filters_list_input[i], self.filters_list_output[i], kernel_size=1, stride=1),
                nn.ReLU()
            )
            fusion_conv_c[i] = nn.Sequential(
                nn.Conv2d(filters_list_input[i], self.filters_list_output[i], kernel_size=1, stride=1),
                nn.ReLU()
            )
        self.fusion_conv_h = nn.ModuleList(fusion_conv_h)
        self.fusion_conv_c = nn.ModuleList(fusion_conv_c)

    def forward(self, h_list, c_list):
        h_concat = [None] * len(self.filters_list_output)
        c_concat = [None] * len(self.filters_list_output)
        # encoder_concat = [None] * len(self.filters_list_output)
        for i in range(len(self.filters_list_output)):
            h_concat[i] = self.fusion_conv_h[i](h_list[i])
            c_concat[i] = self.fusion_conv_c[i](c_list[i])
            # encoder_concat[i] = self.fusion_conv_encoder[i](encoder_output_list[i])
        h_concat = torch.cat(h_concat, dim=1)
        c_concat = torch.cat(c_concat, dim=1)
        # encoder_concat = torch.cat(encoder_concat, dim=1)
        # fusion_output = torch.cat([h_concat, c_concat, encoder_concat], dim=1)
        return h_concat, c_concat


class Decoder_model(nn.Module):
    def __init__(self, pre_frames, ConvLSTM2D_filters, args):
        super(Decoder_model, self).__init__()
        self.args = args
        self.pre_frames = pre_frames
        self.decoder_conv2D = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder_convLSTM2D = ConvLSTM2D(4, ConvLSTM2D_filters, kernel_size=5, img_rowcol=(args.img_width // 2) // 2)
        self.decoder_transconv2D = nn.Sequential(
            nn.ConvTranspose2d(ConvLSTM2D_filters, ConvLSTM2D_filters, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ConvLSTM2D_filters, ConvLSTM2D_filters, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(ConvLSTM2D_filters, 1, kernel_size=1, stride=1)
        )

    def forward(self, frame, h, c):
        pre_frames = [None] * self.pre_frames
        for i in range(self.pre_frames):
            frame = self.decoder_conv2D(frame)
            h, c = self.decoder_convLSTM2D(frame, h, c)
            pre_frames[i] = self.decoder_transconv2D(h)
            frame = torch.sigmoid(pre_frames[i])

        pre_frames = torch.stack(pre_frames, dim=0)
        return pre_frames


class ConvLSTM2D(nn.Module):
    def __init__(self, channels, filters, kernel_size, img_rowcol):
        super(ConvLSTM2D, self).__init__()
        self.filters = filters
        self.padding = kernel_size // 2
        self.conv_x = nn.Conv2d(channels, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=True)
        self.conv_h = nn.Conv2d(filters, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=False)
        self.mul_c = nn.Parameter(torch.zeros([1, filters * 3, img_rowcol, img_rowcol], dtype=torch.float32))

    def forward(self, x, h, c):
        # x -> [batch_size, channels, x, y]
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        i_x, f_x, c_x, o_x = torch.split(x_concat, self.filters, dim=1)
        i_h, f_h, c_h, o_h = torch.split(h_concat, self.filters, dim=1)
        i_c, f_c, o_c = torch.split(self.mul_c, self.filters, dim=1)
        i_t = torch.sigmoid(i_x + i_h + i_c * c)
        f_t = torch.sigmoid(f_x + f_h + f_c * c)
        c_t = torch.tanh(c_x + c_h)
        c_next = i_t * c_t + f_t * c
        o_t = torch.sigmoid(o_x + o_h + o_c * c_next)
        h_next = o_t * torch.tanh(c_next)
        return h_next, c_next


class LightNet_Model(nn.Module):
    def __init__(self, args, obs_tra_frames, obs_channels, wrf_tra_frames, wrf_channels):
        super(LightNet_Model, self).__init__()
        self.args = args
        pre_frames = wrf_tra_frames
        self.encoder_obs_model = Encoder_obs_model(obs_tra_frames, obs_channels, args)
        self.encoder_wrf_model = Encoder_wrf_model(wrf_tra_frames, wrf_channels, args)
        filters_list_output = [16, 8]
        self.fusion_model = Fusion_model([8, 32], filters_list_output, args)   # [obs, wrf]
        self.decoder_model = Decoder_model(pre_frames, sum(filters_list_output), args)

    def forward(self, obs, wrf):
        # encoder
        h_list = []
        c_list = []
        # obs : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        obs = obs.permute(1, 0, 2, 3, 4).contiguous()
        obs_h, obs_c = self.encoder_obs_model(obs)
        h_list.append(obs_h)
        c_list.append(obs_c)

        # wrf : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        wrf = wrf.permute(1, 0, 2, 3, 4).contiguous()
        wrf_h, wrf_c = self.encoder_wrf_model(wrf)
        h_list.append(wrf_h)
        c_list.append(wrf_c)

        # fusion
        fusion_h, fusion_c = self.fusion_model(h_list, c_list)

        # decoder
        pre_frames = self.decoder_model(obs[-1], fusion_h, fusion_c)
        pre_frames = pre_frames.permute(1, 0, 2, 3, 4).contiguous()

        return pre_frames