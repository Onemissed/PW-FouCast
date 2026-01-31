import os
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from util.preprocess import reshape_patch, reshape_patch_back

def calculate_metrics(output, target, thresh, args):
    # Ensure tensors are of float type for calculations
    output, target = output.float(), target.float()

    # Set threshold
    threshold = torch.tensor(thresh).to(args.device)
    output_binary = (output > threshold).float()
    target_binary = (target > threshold).float()

    output_binary[torch.isnan(output_binary)] = 0
    target_binary[torch.isnan(target_binary)] = 0

    # TP
    hits = torch.sum((output_binary == 1) & (target_binary == 1)).float()
    # FN
    misses = torch.sum((output_binary == 0) & (target_binary == 1)).float()
    # FP
    false_alarms = torch.sum((output_binary == 1) & (target_binary == 0)).float()
    # TN
    correct_negatives = torch.sum((output_binary == 0) & (target_binary == 0)).float()

    return hits, misses, false_alarms, correct_negatives


def calculate_scores(metrics):
    """
    Given a dictionary with keys "hits", "misses", "false_alarms", and "correct_negatives",
    compute the metrics: POD, FAR, CSI, and HSS.
    """
    a = metrics["hits"]
    b = metrics["false_alarms"]
    c = metrics["misses"]
    d = metrics["correct_negatives"]

    pod = a / (a + c) if (a + c) > 0 else 0
    far = b / (a + b) if (a + b) > 0 else 0
    csi = a / (a + b + c) if (a + b + c) > 0 else 0
    n = a + b + c + d
    # aref is the reference (expected hits by chance)
    aref = ((a + b) / n * (a + c)) if n > 0 else 0
    denom = (a + b + c - aref)
    gss = (a - aref) / denom if denom != 0 else 0
    hss = 2 * gss / (gss + 1) if (gss + 1) != 0 else 0
    return {"pod": pod, "far": far, "csi": csi, "hss": hss}


class Model_eval(object):
    def __init__(self, args):
        self.args = args
        self.minMSE = 2000
        self.minMSE_epoch = -1
        self.minMAE = 1000
        self.minMAE_epoch = -1
        self.minMSE_MAE = 3000
        self.minMSE_MAE_epoch = -1
        self.maxSSIM = 0
        self.maxSSIM_epoch = -1
        self.maxPSNR = 0
        self.maxPSNR_epoch = -1

        self.maxAvgCSI = -0.5
        self.maxAvgCSI_epoch = -1
        self.maxAvgHSS = -99
        self.maxAvgHSS_epoch = -1

        self.max_metrics = {
            12: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            18: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            24: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            32: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
        }

        self.metrics = {
            12: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            18: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            24: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            32: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
        }

    def eval_update(self, gt, pred, threshold):
        # Calculate the metrics for a given threshold.
        hits, misses, false_alarms, correct_negatives = calculate_metrics(pred, gt, threshold, self.args)
        m = self.metrics[threshold]
        m["hits"] += hits
        m["misses"] += misses
        m["false_alarms"] += false_alarms
        m["correct_negatives"] += correct_negatives

    def eval(self, dataloader, model, epoch):
        mse_loss = 0
        mae_loss = 0
        ssim_total = 0
        psnr_total = 0
        count = 0

        self.metrics = {
            12: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            18: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            24: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            32: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
        }

        mean_pangu = torch.FloatTensor([[1.09202734e+03, 7.45326265e+03, 1.42546326e+04, 2.95425694e+04, 4.13410379e+04,
                                         5.48742997e+04, 7.07713639e+04, 9.01569238e+04, 1.01870097e+05, 1.15904458e+05,
                                         1.33939413e+05, 1.59282060e+05, 2.02378439e+05],
                                        [7.10583028e-03, 5.94396722e-03, 4.60405153e-03, 2.37266590e-03, 1.41726481e-03,
                                         7.77651117e-04, 3.59290469e-04, 1.01829354e-04, 4.28166402e-05, 1.38974410e-05,
                                         4.07840097e-06, 2.82912195e-06, 2.80891535e-06],
                                        [2.85056800e+02, 2.81245819e+02, 2.77635476e+02, 2.69710413e+02, 2.62792075e+02,
                                         2.53858859e+02, 2.42168935e+02, 2.27601840e+02, 2.20932357e+02, 2.18237112e+02,
                                         2.18385517e+02, 2.17027828e+02, 2.16474633e+02],
                                        [1.34955972e+00, 2.60091125e+00, 3.83729798e+00, 5.74944866e+00, 7.14217592e+00,
                                         8.45471305e+00, 9.87351803e+00, 1.13841162e+01, 1.19329809e+01, 1.17900418e+01,
                                         1.00875209e+01, 7.78761681e+00, 3.55718934e+00],
                                        [-5.26488132e-01, 9.27440884e-02, 6.07321239e-01, 1.04833224e+00,
                                         9.64329263e-01,
                                         1.05408965e+00, 8.46995528e-01, 2.46003948e-01,
                                         -5.02358379e-01, -1.20070629e+00, -8.03810358e-01, -1.42929576e-01,
                                         7.29356777e-02]]).to(self.args.device)
        std_pangu = torch.FloatTensor([[7.61526714e+02, 8.02607528e+02, 8.77802495e+02, 1.13150026e+03, 1.36917979e+03,
                                        1.67670899e+03, 2.06480750e+03, 2.51875510e+03, 2.67630123e+03, 2.65650303e+03,
                                        2.50986045e+03, 2.38761726e+03, 2.58569810e+03],
                                       [2.45431549e-03, 2.23532003e-03, 2.17730673e-03, 1.58524609e-03, 1.04406055e-03,
                                        5.91255039e-04, 2.75400905e-04, 8.26091020e-05, 3.51028718e-05, 1.07149253e-05,
                                        1.90020697e-06, 2.65025564e-07, 4.97738354e-08],
                                       [5.71828127e+00, 6.21737757e+00, 6.28582025e+00, 6.39069192e+00, 6.45573319e+00,
                                        6.60594225e+00, 6.51417902e+00, 5.07728371e+00, 3.89614844e+00, 5.12679067e+00,
                                        4.95114556e+00, 3.76351371e+00, 3.95440498e+00],
                                       [5.42642068e+00, 8.17326379e+00, 8.36500974e+00, 9.39924491e+00, 1.08266592e+01,
                                        1.26890762e+01, 1.50737278e+01, 1.75978958e+01, 1.76148925e+01, 1.51843940e+01,
                                        1.09858047e+01, 8.67750218e+00, 1.07300057e+01],
                                       [5.15005319e+00, 7.13411336e+00, 7.03851510e+00, 8.21251796e+00, 9.66016525e+00,
                                        1.17675720e+01, 1.48828849e+01, 1.82024520e+01, 1.84245086e+01, 1.48993595e+01,
                                        9.64625118e+00, 6.41299913e+00, 4.43226793e+00]]).to(self.args.device)
        mean_pangu = mean_pangu.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        std_pangu = std_pangu.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            with tqdm(total=len(dataloader)) as pbar:
                for i, (X, X_pangu) in enumerate(dataloader):
                    ims = X.numpy()
                    ims_pangu = X_pangu.float().to(self.args.device)
                    if self.args.model in ['predrnn_v2', 'mm_rnn']:
                        ims_convlstm = X.numpy().astype(np.float32)
                        ims_convlstm = np.expand_dims(ims_convlstm, axis=4)
                        ims /= 80
                        ims_convlstm = reshape_patch(ims_convlstm, self.args.patch_size)
                        ims_convlstm = torch.FloatTensor(ims_convlstm).to(self.args.device)
                        if self.args.model == 'predrnn_v2':
                            # Reverse scheduled sampling
                            real_input_flag = torch.zeros(
                                (self.args.batchsize,
                                 self.args.total_length - 2,
                                 self.args.img_width // self.args.patch_size,
                                 self.args.img_width // self.args.patch_size,
                                 self.args.patch_size ** 2 * self.args.img_channel))
                            real_input_flag[:, :self.args.input_length - 1, :, :] = 1.0
                            real_input_flag = real_input_flag.to(self.args.device)
                        elif self.args.model == 'mm_rnn':
                            # Scheduled sampling
                            real_input_flag_mm = torch.zeros(
                                (self.args.batchsize,
                                 self.args.total_length - self.args.input_length - 1,
                                 self.args.img_width // self.args.patch_size,
                                 self.args.img_width // self.args.patch_size,
                                 self.args.patch_size ** 2 * self.args.img_channel))
                            real_input_flag_mm = real_input_flag_mm.to(self.args.device)

                    target = ims[:, :, :, 5:]
                    ims = torch.FloatTensor(ims).to(self.args.device)

                    ims /= 80
                    if self.args.model in ['earthformer', 'nowcastnet']:
                        # For earthformer, change the tensor shape to [B, T, H, W, C]
                        ims = ims.unsqueeze(dim=4)
                    else:
                        # [B, T, C, H, W]
                        ims = ims.unsqueeze(dim=2)

                    B, T, var, pressure_level, H, W = ims_pangu.shape
                    # Perform channel-wise normalization on Pangu-Weather forecasts
                    ims_pangu = (ims_pangu - mean_pangu) / std_pangu
                    ims_pangu = ims_pangu[:, :, :, 2:6, :, :]
                    ims_pangu = ims_pangu.reshape(B, T, -1, H, W)

                    if self.args.model == 'pw_foucast':
                        pred = model(ims[:, :5], ims_pangu, None)
                    elif self.args.model in ['earthformer', 'nowcastnet', 'afno']:
                        pred = model(ims[:, :5])
                    elif self.args.model in ['simvp_v2', 'tau', 'pastnet']:
                        pred_1 = model(ims[:, :5])
                        pred_2 = model(pred_1)
                        pred_3 = model(pred_2)
                        pred_4 = model(pred_3)
                        pred = torch.cat((pred_1, pred_2, pred_3, pred_4), dim=1)
                    elif self.args.model == 'alphapre':
                        pred, _ = model.module.predict(frames_in=ims[:, :5], frames_gt=ims[:, 5:], compute_loss=False)
                    elif self.args.model == 'lmc_memory':
                        # define data indexes
                        short_start, short_end = 0, self.args.short_len
                        short_data = ims[:, short_start:short_end, :, :, :]
                        pred = model(short_data, None, self.args.out_len, phase=2)
                    elif self.args.model in ['lightnet', 'stjointnet']:
                        pred = model(ims[:, :5], ims_pangu)
                    elif self.args.model == 'predrnn_v2':
                        pred1, _ = model(ims_convlstm, real_input_flag)
                    elif self.args.model == 'mm_rnn':
                        pred = model(ims_convlstm, ims_pangu, real_input_flag_mm)

                    pred *= 255
                    pred.clamp_(min=0, max=255)

                    if self.args.model in ['predrnn_v2', 'mm_rnn']:
                        pred = reshape_patch_back(pred.cpu().numpy(), self.args.patch_size)
                        if self.args.model == 'predrnn_v2':
                            pred = pred[:, -self.args.output_length:]

                    if self.args.model in ['earthformer', 'nowcastnet', 'predrnn_v2', 'mm_rnn']:
                        img_out = pred.squeeze(dim=4)
                    else:
                        img_out = pred.squeeze(dim=2)
                    img_out = img_out.cpu().numpy()

                    mse = np.mean(np.square(target - img_out))
                    mae = np.mean(np.abs(target - img_out))

                    ssim_temp = 0.0
                    psnr_temp = 0.0
                    epsilon = 1e-10
                    for b in range(B):
                        for f in range(T):
                            # Remove the channel dimension as `ssim` expects 2D images
                            output_frame = img_out[b, f, :, :]
                            target_frame = target[b, f, :, :]

                            # Compute SSIM for the single frame pair and add to the total
                            ssim_value = ssim(output_frame, target_frame, data_range=255.0)
                            ssim_temp += ssim_value

                            mse_temp = np.mean((output_frame - target_frame) ** 2)
                            psnr_value = 20 * np.log10(255 / np.sqrt(mse_temp + epsilon))
                            psnr_temp += psnr_value

                    ssim_mean = ssim_temp / (B * T)
                    psnr_mean = psnr_temp / (B * T)
                    ssim_total += ssim_mean
                    psnr_total += psnr_mean

                    mse_loss = mse_loss + mse
                    mae_loss = mae_loss + mae
                    count = count + 1

                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=12)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=18)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=24)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=32)

                    pbar.update(1)

            mse_score = mse_loss / count
            mae_score = mae_loss / count
            ssim_score = ssim_total / count
            psnr_score = psnr_total / count

            scores = {}
            for thresh, m in self.metrics.items():
                scores[thresh] = calculate_scores(m)

            info = 'Test EPOCH INFO: epoch:{} \nMSE:{:.4f}  MAE:{:.4f}  SSIM:{:.4f}  PSNR:{:.4f}  MSE_MAE:{:.4f}\nCSI_12:{:.4f}  CSI_18:{:.4f}  CSI_24:{:.4f}  CSI_32:{:.4f}\nHSS_12:{:.4f}  HSS_18:{:.4f}  HSS_24:{:.4f}  HSS_32:{:.4f}\n'. \
                format(epoch + 1, mse_score, mae_score, ssim_score, psnr_score, mse_score + mae_score,
                       scores[12]['csi'], scores[18]['csi'], scores[24]['csi'], scores[32]['csi'],
                       scores[12]['hss'], scores[18]['hss'], scores[24]['hss'], scores[32]['hss'])
            print(info)

            if mse_score < self.minMSE:
                self.minMSE = mse_score
                self.minMSE_epoch = epoch + 1
            if mae_score < self.minMAE:
                self.minMAE = mae_score
                self.minMAE_epoch = epoch + 1
            if mse_score + mae_score < self.minMSE_MAE:
                self.minMSE_MAE = mse_score + mae_score
                self.minMSE_MAE_epoch = epoch + 1
            if ssim_score > self.maxSSIM:
                self.maxSSIM = ssim_score
                self.maxSSIM_epoch = epoch + 1
            if psnr_score > self.maxPSNR and psnr_score < 1000:
                self.maxPSNR = psnr_score
                self.maxPSNR_epoch = epoch + 1
            if scores[12]['csi'] > self.max_metrics[12]['maxCSI']:
                self.max_metrics[12]['maxCSI'] = scores[12]['csi']
                self.max_metrics[12]['maxCSI_epoch'] = epoch + 1
            if scores[12]['hss'] > self.max_metrics[12]['maxHSS']:
                self.max_metrics[12]['maxHSS'] = scores[12]['hss']
                self.max_metrics[12]['maxHSS_epoch'] = epoch + 1

            if scores[18]['csi'] > self.max_metrics[18]['maxCSI']:
                self.max_metrics[18]['maxCSI'] = scores[18]['csi']
                self.max_metrics[18]['maxCSI_epoch'] = epoch + 1
            if scores[18]['hss'] > self.max_metrics[18]['maxHSS']:
                self.max_metrics[18]['maxHSS'] = scores[18]['hss']
                self.max_metrics[18]['maxHSS_epoch'] = epoch + 1

            if scores[24]['csi'] > self.max_metrics[24]['maxCSI']:
                self.max_metrics[24]['maxCSI'] = scores[24]['csi']
                self.max_metrics[24]['maxCSI_epoch'] = epoch + 1
            if scores[24]['hss'] > self.max_metrics[24]['maxHSS']:
                self.max_metrics[24]['maxHSS'] = scores[24]['hss']
                self.max_metrics[24]['maxHSS_epoch'] = epoch + 1

            if scores[32]['csi'] > self.max_metrics[32]['maxCSI']:
                self.max_metrics[32]['maxCSI'] = scores[32]['csi']
                self.max_metrics[32]['maxCSI_epoch'] = epoch + 1
            if scores[32]['hss'] > self.max_metrics[32]['maxHSS']:
                self.max_metrics[32]['maxHSS'] = scores[32]['hss']
                self.max_metrics[32]['maxHSS_epoch'] = epoch + 1

            avgcsi = (scores[12]['csi'] + scores[18]['csi'] + scores[24]['csi'] + scores[32]['csi']) / 4
            avghss = (scores[12]['hss'] + scores[18]['hss'] + scores[24]['hss'] + scores[32]['hss']) / 4
            if avgcsi > self.maxAvgCSI:
                self.maxAvgCSI = avgcsi
                self.maxAvgCSI_epoch = epoch + 1
            if avghss > self.maxAvgHSS:
                self.maxAvgHSS = avghss
                self.maxAvgHSS_epoch = epoch + 1

            print(
                "minMSE: {:.4f}  epoch:{}\nminMAE: {:.4f}  epoch:{}\nminMSE_MAE: {:.4f}  epoch:{}\nmaxSSIM: {:.4f}  epoch:{}\nmaxPSNR: {:.4f}  epoch:{}\nmaxCSI_12: {:.4f}  epoch:{}\nmaxCSI_18: {:.4f}  epoch:{}\nmaxCSI_24: {:.4f}  epoch:{}\nmaxCSI_32: {:.4f}  epoch:{}\nmaxHSS_12: {:.4f}  epoch:{}\nmaxHSS_18: {:.4f}  epoch:{}\nmaxHSS_24: {:.4f}  epoch:{}\nmaxHSS_32: {:.4f}  epoch:{}\nmaxAvgCSI: {:.4f}  epoch:{}\nmaxAvgHSS: {:.4f}  epoch:{}\n".format(
                    self.minMSE, self.minMSE_epoch, self.minMAE, self.minMAE_epoch, self.minMSE_MAE,
                    self.minMSE_MAE_epoch, self.maxSSIM, self.maxSSIM_epoch, self.maxPSNR, self.maxPSNR_epoch,
                    self.max_metrics[12]['maxCSI'], self.max_metrics[12]['maxCSI_epoch'],
                    self.max_metrics[18]['maxCSI'], self.max_metrics[18]['maxCSI_epoch'],
                    self.max_metrics[24]['maxCSI'], self.max_metrics[24]['maxCSI_epoch'],
                    self.max_metrics[32]['maxCSI'], self.max_metrics[32]['maxCSI_epoch'],
                    self.max_metrics[12]['maxHSS'], self.max_metrics[12]['maxHSS_epoch'],
                    self.max_metrics[18]['maxHSS'], self.max_metrics[18]['maxHSS_epoch'],
                    self.max_metrics[24]['maxHSS'], self.max_metrics[24]['maxHSS_epoch'],
                    self.max_metrics[32]['maxHSS'], self.max_metrics[32]['maxHSS_epoch'],
                    self.maxAvgCSI, self.maxAvgCSI_epoch, self.maxAvgHSS, self.maxAvgHSS_epoch))

            with open(os.path.join(self.args.record_dir, 'record.txt'), 'a') as f:
                f.write(info)
                f.write(f"Avg_CSI: {avgcsi:.4f}\tAvg_HSS: {avghss:.4f}\n\n")
                if epoch + 1 == self.args.epoch:
                    f.write(
                        "minMSE: {:.4f}  epoch:{}\nminMAE: {:.4f}  epoch:{}\nminMSE_MAE: {:.4f}  epoch:{}\nmaxSSIM: {:.4f}  epoch:{}\nmaxPSNR: {:.4f}  epoch:{}\nmaxCSI_12: {:.4f}  epoch:{}\nmaxCSI_18: {:.4f}  epoch:{}\nmaxCSI_24: {:.4f}  epoch:{}\nmaxCSI_32: {:.4f}  epoch:{}\nmaxHSS_12: {:.4f}  epoch:{}\nmaxHSS_18: {:.4f}  epoch:{}\nmaxHSS_24: {:.4f}  epoch:{}\nmaxHSS_32: {:.4f}  epoch:{}\nmaxAvgCSI: {:.4f}  epoch:{}\nmaxAvgHSS: {:.4f}  epoch:{}\n".format(
                            self.minMSE, self.minMSE_epoch, self.minMAE, self.minMAE_epoch, self.minMSE_MAE,
                            self.minMSE_MAE_epoch, self.maxSSIM, self.maxSSIM_epoch, self.maxPSNR, self.maxPSNR_epoch,
                            self.max_metrics[12]['maxCSI'], self.max_metrics[12]['maxCSI_epoch'],
                            self.max_metrics[18]['maxCSI'], self.max_metrics[18]['maxCSI_epoch'],
                            self.max_metrics[24]['maxCSI'], self.max_metrics[24]['maxCSI_epoch'],
                            self.max_metrics[32]['maxCSI'], self.max_metrics[32]['maxCSI_epoch'],
                            self.max_metrics[12]['maxHSS'], self.max_metrics[12]['maxHSS_epoch'],
                            self.max_metrics[18]['maxHSS'], self.max_metrics[18]['maxHSS_epoch'],
                            self.max_metrics[24]['maxHSS'], self.max_metrics[24]['maxHSS_epoch'],
                            self.max_metrics[32]['maxHSS'], self.max_metrics[32]['maxHSS_epoch'],
                            self.maxAvgCSI, self.maxAvgCSI_epoch, self.maxAvgHSS, self.maxAvgHSS_epoch))