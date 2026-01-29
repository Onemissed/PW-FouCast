import os
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

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


class Model_eval_nonRNN(object):
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
            16: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            74: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            133: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            160: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            181: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
            219: {"maxCSI": -0.5, "maxCSI_epoch": -1, "maxHSS": -99, "maxHSS_epoch": -1},
        }
        # self.thresholds = dBZ_to_pixel(np.array([5.0, 20.0, 40.0]))

        self.metrics = {
            16: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            74: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            133: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            160: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            181: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            219: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
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
            16: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            74: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            133: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            160: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            181: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
            219: {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0},
        }

        with torch.no_grad():
            with tqdm(total=len(dataloader)) as pbar:
                for i, X in enumerate(dataloader):
                    ims = X.numpy()
                    target = ims[:, :, :, 7:]
                    target = np.transpose(target, (0, 3, 1, 2))
                    B, T, H, W = target.shape
                    ims = torch.FloatTensor(ims).to(self.args.device)
                    ims = ims.permute(0, 3, 1, 2)

                    # Normalization
                    ims /= 255
                    if self.args.model == 'earthformer':
                        # [B, T, H, W, C] for earthformer
                        ims = ims.unsqueeze(dim=4)
                    else:
                        # [B, T, C, H, W]
                        ims = ims.unsqueeze(dim=2)

                    pred = model(ims[:, :7])
                    pred = pred[:, :6]
                    # Denorm
                    pred *= 255
                    pred.clamp_(min=0, max=255)

                    if self.args.model == 'earthformer':
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

                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=16)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=74)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=133)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=160)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=181)
                    self.eval_update(torch.from_numpy(target).to(self.args.device), torch.from_numpy(img_out).to(self.args.device), threshold=219)

                    pbar.update(1)

            mse_score = mse_loss / count
            mae_score = mae_loss / count
            ssim_score = ssim_total / count
            psnr_score = psnr_total / count

            scores = {}
            for thresh, m in self.metrics.items():
                scores[thresh] = calculate_scores(m)

            info = 'Test EPOCH INFO: epoch:{} \nMSE:{:.4f}  MAE:{:.4f}  SSIM:{:.4f}  PSNR:{:.4f}  MSE_MAE:{:.4f}\nCSI_16:{:.4f}  CSI_74:{:.4f}  CSI_133:{:.4f}  CSI_160:{:.4f}  CSI_181:{:.4f}  CSI_219:{:.4f}\nHSS_16:{:.4f}  HSS_74:{:.4f}  HSS_133:{:.4f}  HSS_160:{:.4f}  HSS_181:{:.4f}  HSS_219:{:.4f}\n'. \
                format(epoch + 1, mse_score, mae_score, ssim_score, psnr_score, mse_score + mae_score,
                       scores[16]['csi'], scores[74]['csi'], scores[133]['csi'], scores[160]['csi'], scores[181]['csi'], scores[219]['csi'],
                       scores[16]['hss'], scores[74]['hss'], scores[133]['hss'], scores[160]['hss'], scores[181]['hss'], scores[219]['hss'])
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

            if scores[16]['csi'] > self.max_metrics[16]['maxCSI']:
                self.max_metrics[16]['maxCSI'] = scores[16]['csi']
                self.max_metrics[16]['maxCSI_epoch'] = epoch + 1
            if scores[16]['hss'] > self.max_metrics[16]['maxHSS']:
                self.max_metrics[16]['maxHSS'] = scores[16]['hss']
                self.max_metrics[16]['maxHSS_epoch'] = epoch + 1

            if scores[74]['csi'] > self.max_metrics[74]['maxCSI']:
                self.max_metrics[74]['maxCSI'] = scores[74]['csi']
                self.max_metrics[74]['maxCSI_epoch'] = epoch + 1
            if scores[74]['hss'] > self.max_metrics[74]['maxHSS']:
                self.max_metrics[74]['maxHSS'] = scores[74]['hss']
                self.max_metrics[74]['maxHSS_epoch'] = epoch + 1

            if scores[133]['csi'] > self.max_metrics[133]['maxCSI']:
                self.max_metrics[133]['maxCSI'] = scores[133]['csi']
                self.max_metrics[133]['maxCSI_epoch'] = epoch + 1
            if scores[133]['hss'] > self.max_metrics[133]['maxHSS']:
                self.max_metrics[133]['maxHSS'] = scores[133]['hss']
                self.max_metrics[133]['maxHSS_epoch'] = epoch + 1

            if scores[160]['csi'] > self.max_metrics[160]['maxCSI']:
                self.max_metrics[160]['maxCSI'] = scores[160]['csi']
                self.max_metrics[160]['maxCSI_epoch'] = epoch + 1
            if scores[160]['hss'] > self.max_metrics[160]['maxHSS']:
                self.max_metrics[160]['maxHSS'] = scores[160]['hss']
                self.max_metrics[160]['maxHSS_epoch'] = epoch + 1

            if scores[181]['csi'] > self.max_metrics[181]['maxCSI']:
                self.max_metrics[181]['maxCSI'] = scores[181]['csi']
                self.max_metrics[181]['maxCSI_epoch'] = epoch + 1
            if scores[181]['hss'] > self.max_metrics[181]['maxHSS']:
                self.max_metrics[181]['maxHSS'] = scores[181]['hss']
                self.max_metrics[181]['maxHSS_epoch'] = epoch + 1

            if scores[219]['csi'] > self.max_metrics[219]['maxCSI']:
                self.max_metrics[219]['maxCSI'] = scores[219]['csi']
                self.max_metrics[219]['maxCSI_epoch'] = epoch + 1
            if scores[219]['hss'] > self.max_metrics[219]['maxHSS']:
                self.max_metrics[219]['maxHSS'] = scores[219]['hss']
                self.max_metrics[219]['maxHSS_epoch'] = epoch + 1

            avgcsi = (scores[16]['csi'] + scores[74]['csi'] + scores[133]['csi'] + scores[160]['csi'] + scores[181]['csi'] + scores[219]['csi']) / 6
            avghss = (scores[16]['hss'] + scores[74]['hss'] + scores[133]['hss'] + scores[160]['hss'] + scores[181]['hss'] + scores[219]['hss']) / 6
            if avgcsi > self.maxAvgCSI:
                self.maxAvgCSI = avgcsi
                self.maxAvgCSI_epoch = epoch + 1
            if avghss > self.maxAvgHSS:
                self.maxAvgHSS = avghss
                self.maxAvgHSS_epoch = epoch + 1

            print(
                "minMSE: {:.4f}  epoch:{}\nminMAE: {:.4f}  epoch:{}\nminMSE_MAE: {:.4f}  epoch:{}\nmaxSSIM: {:.4f}  epoch:{}\nmaxPSNR: {:.4f}  epoch:{}\nmaxCSI_16: {:.4f}  epoch:{}\nmaxCSI_74: {:.4f}  epoch:{}\nmaxCSI_133: {:.4f}  epoch:{}\nmaxCSI_160: {:.4f}  epoch:{}\nmaxCSI_181: {:.4f}  epoch:{}\nmaxCSI_219: {:.4f}  epoch:{}\nmaxHSS_16: {:.4f}  epoch:{}\nmaxHSS_74: {:.4f}  epoch:{}\nmaxHSS_133: {:.4f}  epoch:{}\nmaxHSS_160: {:.4f}  epoch:{}\nmaxHSS_181: {:.4f}  epoch:{}\nmaxHSS_219: {:.4f}  epoch:{}\nmaxAvgCSI: {:.4f}  epoch:{}\nmaxAvgHSS: {:.4f}  epoch:{}\n".format(
                    self.minMSE, self.minMSE_epoch, self.minMAE, self.minMAE_epoch, self.minMSE_MAE, self.minMSE_MAE_epoch, self.maxSSIM, self.maxSSIM_epoch, self.maxPSNR, self.maxPSNR_epoch,
                    self.max_metrics[16]['maxCSI'], self.max_metrics[16]['maxCSI_epoch'], self.max_metrics[74]['maxCSI'], self.max_metrics[74]['maxCSI_epoch'], self.max_metrics[133]['maxCSI'], self.max_metrics[133]['maxCSI_epoch'],
                    self.max_metrics[160]['maxCSI'], self.max_metrics[160]['maxCSI_epoch'], self.max_metrics[181]['maxCSI'], self.max_metrics[181]['maxCSI_epoch'], self.max_metrics[219]['maxCSI'], self.max_metrics[219]['maxCSI_epoch'],
                    self.max_metrics[16]['maxHSS'], self.max_metrics[16]['maxHSS_epoch'], self.max_metrics[74]['maxHSS'], self.max_metrics[74]['maxHSS_epoch'], self.max_metrics[133]['maxHSS'], self.max_metrics[133]['maxHSS_epoch'],
                    self.max_metrics[160]['maxHSS'], self.max_metrics[160]['maxHSS_epoch'], self.max_metrics[181]['maxHSS'], self.max_metrics[181]['maxHSS_epoch'], self.max_metrics[219]['maxHSS'], self.max_metrics[219]['maxHSS_epoch'],
                    self.maxAvgCSI, self.maxAvgCSI_epoch, self.maxAvgHSS, self.maxAvgHSS_epoch))

            # Save evaluation results
            with open(os.path.join(self.args.record_dir, 'record.txt'), 'a') as f:
                f.write(info)
                f.write(f"Avg_CSI: {avgcsi:.4f}\tAvg_HSS: {avghss:.4f}\n\n")
                if epoch + 1 == self.args.epoch:
                    f.write(
                        "minMSE: {:.4f}  epoch:{}\nminMAE: {:.4f}  epoch:{}\nminMSE_MAE: {:.4f}  epoch:{}\nmaxSSIM: {:.4f}  epoch:{}\nmaxPSNR: {:.4f}  epoch:{}\nmaxCSI_16: {:.4f}  epoch:{}\nmaxCSI_74: {:.4f}  epoch:{}\nmaxCSI_133: {:.4f}  epoch:{}\nmaxCSI_160: {:.4f}  epoch:{}\nmaxCSI_181: {:.4f}  epoch:{}\nmaxCSI_219: {:.4f}  epoch:{}\nmaxHSS_16: {:.4f}  epoch:{}\nmaxHSS_74: {:.4f}  epoch:{}\nmaxHSS_133: {:.4f}  epoch:{}\nmaxHSS_160: {:.4f}  epoch:{}\nmaxHSS_181: {:.4f}  epoch:{}\nmaxHSS_219: {:.4f}  epoch:{}\nmaxAvgCSI: {:.4f}  epoch:{}\nmaxAvgHSS: {:.4f}  epoch:{}\n".format(
                            self.minMSE, self.minMSE_epoch, self.minMAE, self.minMAE_epoch, self.minMSE_MAE, self.minMSE_MAE_epoch, self.maxSSIM, self.maxSSIM_epoch, self.maxPSNR, self.maxPSNR_epoch,
                            self.max_metrics[16]['maxCSI'], self.max_metrics[16]['maxCSI_epoch'], self.max_metrics[74]['maxCSI'], self.max_metrics[74]['maxCSI_epoch'], self.max_metrics[133]['maxCSI'], self.max_metrics[133]['maxCSI_epoch'],
                            self.max_metrics[160]['maxCSI'], self.max_metrics[160]['maxCSI_epoch'], self.max_metrics[181]['maxCSI'], self.max_metrics[181]['maxCSI_epoch'], self.max_metrics[219]['maxCSI'], self.max_metrics[219]['maxCSI_epoch'],
                            self.max_metrics[16]['maxHSS'], self.max_metrics[16]['maxHSS_epoch'], self.max_metrics[74]['maxHSS'], self.max_metrics[74]['maxHSS_epoch'], self.max_metrics[133]['maxHSS'], self.max_metrics[133]['maxHSS_epoch'],
                            self.max_metrics[160]['maxHSS'], self.max_metrics[160]['maxHSS_epoch'], self.max_metrics[181]['maxHSS'], self.max_metrics[181]['maxHSS_epoch'], self.max_metrics[219]['maxHSS'], self.max_metrics[219]['maxHSS_epoch'],
                            self.maxAvgCSI, self.maxAvgCSI_epoch, self.maxAvgHSS, self.maxAvgHSS_epoch))