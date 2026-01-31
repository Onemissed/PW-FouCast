import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from generator_meteonet_withpangu import DataGenerator
from model.pw_foucast import PW_FouCast
from evaluation.scores_meteonet import Model_eval

def load_model_config(model_name: str, cfg_root='config/sevir') -> dict:
    path = os.path.join(cfg_root, f'{model_name.lower()}.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config for model '{model_name}' not found at {path!r}")
    with open(path, 'r') as f:
        full_cfg = yaml.safe_load(f)
    return full_cfg['model']

def DoTrain(args):
    # Data index
    TrainSetFilePath = './data_index/meteonet_train_periods.txt'
    TestSetFilePath = './data_index/meteonet_test_periods.txt'
    train_list = []
    with open(TrainSetFilePath) as file:
        for line in file:
            train_list.append(line.rstrip('\n').rstrip('\r\n'))
    test_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            test_list.append(line.rstrip('\n').rstrip('\r\n'))

    # data
    train_data = DataGenerator(train_list)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
    test_data = DataGenerator(test_list)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False, num_workers=0)

    MODEL_REGISTRY = {
        'pw_foucast': PW_FouCast,
    }
    ModelClass = MODEL_REGISTRY.get(args.model.lower())
    if ModelClass is None:
        raise ValueError(f'Unknown model: {args.model!r}. '
                         f'Available models: {list(MODEL_REGISTRY)}')

    # Load the model configuration
    model_kwargs = load_model_config(args.model.lower(), f'config/{args.dataset}')
    model_kwargs['args'] = args
    print("dataset: ", args.dataset)
    print("model: ", args.model)
    print("lr: ", args.lr)
    print("batch size: ", args.batchsize)
    print("using gpu: ", args.gpus)
    model = ModelClass(**model_kwargs).to(args.device)
    model = torch.nn.DataParallel(model)

    # Loss function
    MSE_criterion = nn.MSELoss(reduction='mean')

    # Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = 2 * len(train_loader) * args.epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)

    # eval
    model_eval_testdata = Model_eval(args)

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
                                    [-5.26488132e-01, 9.27440884e-02, 6.07321239e-01, 1.04833224e+00, 9.64329263e-01,
                                     1.05408965e+00, 8.46995528e-01, 2.46003948e-01,
                                     -5.02358379e-01, -1.20070629e+00, -8.03810358e-01, -1.42929576e-01,
                                     7.29356777e-02]]).to(args.device)
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
                                    9.64625118e+00, 6.41299913e+00, 4.43226793e+00]]).to(args.device)
    mean_pangu = mean_pangu.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
    std_pangu = std_pangu.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)

    print('>'*35 + ' training ' + '<'*35)
    for epoch in range(args.epoch):
        print("epoch:", epoch + 1)
        with tqdm(total=len(train_loader)) as pbar:
            model.train()
            for i, (X, X_pangu) in enumerate(train_loader):
                ims = X.float().to(args.device)
                ims_pangu = X_pangu.float().to(args.device)
                # Normalization
                ims /= 80
                # Shape: [B, T, C, H, W]
                ims = ims.unsqueeze(dim=2)
                B, T, var, pressure_level, H, W = ims_pangu.shape

                # Normalize Pangu-Weather forecasts
                ims_pangu = (ims_pangu - mean_pangu) / std_pangu

                ims_pangu = ims_pangu[:, :, :, 2:6, :, :]
                ims_pangu = ims_pangu.reshape(B, T, -1, H, W)

                """Phase 1: Storing stage"""
                model.module.memory.phase_memory.requires_grad = True
                pred = model(ims[:, :5], ims_pangu, ims[:, 5:])

                pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
                gt_freq = torch.fft.fft2(ims[:, 5:], dim=(-2, -1))
                freq_loss1 = torch.mean(torch.abs(pred_freq - gt_freq))

                loss = MSE_criterion(pred, ims[:, 5:]) + 0.55 * freq_loss1

                # backward
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()
                scheduler.step()

                """Phase 2: Matching stage"""
                model.module.memory.phase_memory.requires_grad = False
                pred = model(ims[:, :5], ims_pangu, None)

                pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
                gt_freq = torch.fft.fft2(ims[:, 5:], dim=(-2, -1))
                freq_loss2 = torch.mean(torch.abs(pred_freq - gt_freq))

                loss = MSE_criterion(pred, ims[:, 5:]) + 0.55 * freq_loss2

                # backward
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()
                scheduler.step()

                lr = optimizer.state_dict()['param_groups'][0]['lr']
                pbar.set_description('train loss: {:.5f}, learning rate: {:.7f}'.format(loss.item(), lr))
                pbar.update(1)

        model_weights_path = args.model_weight_dir + f'model_weight_epoch_{epoch + 1}.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            # 'loss': loss,
        }
        torch.save(checkpoint, model_weights_path)

        model.eval()
        model_eval_testdata.eval(test_loader, model, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pw_foucast', help='The model used for training')
    parser.add_argument('--dataset', type=str, default='meteonet', help='dataset name')
    parser.add_argument('--patch_size', type=int, default=4, help='')
    parser.add_argument('--img_width', type=int, default=128, help='image size')
    parser.add_argument('--img_channel', type=int, default=1, help='')

    # Training hyperparameters
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='use cpu/gpu')
    parser.add_argument('--gpus', type=str, default='0,1', help='gpu device ID')
    parser.add_argument('--epoch', type=int, default=100, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_length', type=int, default=5, help='')
    parser.add_argument('--output_length', type=int, default=20, help='')
    parser.add_argument('--total_length', type=int, default=25, help='')
    parser.add_argument('--record_dir', type=str, default=None, help='')
    parser.add_argument('--model_weight_dir', type=str, default=None, help='')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    current_file = args.dataset + '/' + args.model
    # Create directory
    folder_path = './record/' + current_file
    weight_path = './model_weight/' + current_file
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    args.record_dir = folder_path + '/'
    args.model_weight_dir = weight_path + '/'

    DoTrain(args)
