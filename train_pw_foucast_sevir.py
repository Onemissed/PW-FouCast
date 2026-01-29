import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from generator_sevir_withpangu import DataGenerator
from model.pw_foucast import PW_FouCast
from evaluation.scores_sevir import Model_eval

def load_model_config(model_name: str, cfg_root='config/sevir') -> dict:
    path = os.path.join(cfg_root, f'{model_name.lower()}.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config for model '{model_name}' not found at {path!r}")
    with open(path, 'r') as f:
        full_cfg = yaml.safe_load(f)
    return full_cfg['model']

def DoTrain(args):
    # Data index
    TrainSetFilePath = './data_index/sevir_train_periods.txt'
    TestSetFilePath = './data_index/sevir_test_periods.txt'
    Train_pangu = './data_index/sevir_train_pangufile.txt'
    Test_pangu = './data_index/sevir_test_pangufile.txt'
    train_list = []
    with open(TrainSetFilePath) as file:
        for line in file:
            train_list.append(line.rstrip('\n').rstrip('\r\n'))
    test_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            test_list.append(line.rstrip('\n').rstrip('\r\n'))

    train_list_pangu = []
    with open(Train_pangu) as file:
        for line in file:
            train_list_pangu.append(line.rstrip('\n').rstrip('\r\n'))
    test_list_pangu = []
    with open(Test_pangu) as file:
        for line in file:
            test_list_pangu.append(line.rstrip('\n').rstrip('\r\n'))

    # data
    train_data = DataGenerator(train_list, train_list_pangu)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
    test_data = DataGenerator(test_list, test_list_pangu)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False, num_workers=0)

    MODEL_REGISTRY = {
        'pw_foucast': PW_FouCast,
    }
    ModelClass = MODEL_REGISTRY.get(args.model.lower())
    if ModelClass is None:
        raise ValueError(f'Unknown model: {args.model!r}. '
                         f'Available models: {list(MODEL_REGISTRY)}')

    # Load the model configuration
    model_kwargs = load_model_config(args.model.lower(), 'config/sevir')
    model_kwargs['args'] = args
    print("model: ", args.model)
    print("lr: ", args.lr)
    print("batch size: ", args.batchsize)
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

    mean_pangu = torch.FloatTensor([[1.2078049e+03, 7.7082539e+03, 1.4697904e+04, 3.0432422e+04, 4.2541539e+04,
                                     5.6434527e+04, 7.2791742e+04, 9.2798367e+04, 1.0485328e+05, 1.1906316e+05,
                                     1.3682492e+05, 1.6127825e+05, 2.0300519e+05],
                                    [1.0179849e-02, 9.2471214e-03, 7.9356497e-03, 4.6562399e-03, 2.9685164e-03,
                                     1.6178534e-03, 7.2102470e-04, 2.2675630e-04, 9.6664808e-05, 3.0249636e-05,
                                     7.1989612e-06, 3.2537953e-06, 2.7416634e-06],
                                    [2.9258209e+02, 2.8883163e+02, 2.8549234e+02, 2.7683734e+02, 2.6940411e+02,
                                     2.6073038e+02, 2.4961644e+02, 2.3479866e+02, 2.2618695e+02, 2.1818420e+02,
                                     2.1256454e+02, 2.0826337e+02, 2.1220099e+02],
                                    [-2.8039768e-02, 1.1677972e+00, 3.8014123e+00, 7.9972882e+00, 1.0487517e+01,
                                     1.3162412e+01, 1.6313780e+01, 2.0241735e+01, 2.2772217e+01, 2.5081093e+01,
                                     2.3105591e+01, 1.4192363e+01, 1.9180852e+00],
                                    [7.4789953e-01, 2.4125929e+00, 3.0797243e+00, 3.1701045e+00, 3.3532467e+00,
                                     3.9827416e+00, 4.9025025e+00, 6.2204680e+00, 6.7110858e+00, 5.9333363e+00,
                                     3.2419741e+00, 1.5795683e+00, -2.0848401e-01]]).to(args.device)
    std_pangu = torch.FloatTensor([[5.28080444e+02, 5.36451294e+02, 6.16971985e+02, 9.07529907e+02, 1.15491357e+03,
                                    1.44350842e+03, 1.81731311e+03, 2.28641309e+03, 2.51649658e+03, 2.63734521e+03,
                                    2.48090820e+03, 2.02926318e+03, 2.03503564e+03],
                                   [5.15696546e-03, 4.53283638e-03, 3.84348631e-03, 2.32248916e-03, 1.74502947e-03,
                                    1.15029421e-03, 5.80821652e-04, 1.88310587e-04, 7.83048090e-05, 2.17211364e-05,
                                    4.49440222e-06, 1.08677830e-06, 7.71736026e-08],
                                   [9.64100552e+00, 9.25792503e+00, 8.21685886e+00, 6.71069002e+00, 6.02426720e+00,
                                    6.08346653e+00, 6.30912685e+00, 5.69921684e+00, 4.68841171e+00, 4.17678738e+00,
                                    5.23956919e+00, 5.79260349e+00, 3.86870313e+00],
                                   [3.20758176e+00, 6.28128672e+00, 7.13523054e+00, 8.08505726e+00, 9.27165413e+00,
                                    1.09642000e+01, 1.31593838e+01, 1.60287724e+01, 1.71699276e+01, 1.76054688e+01,
                                    1.53138847e+01, 1.23805513e+01, 9.86994743e+00],
                                   [3.64704537e+00, 6.83500910e+00, 7.40714741e+00, 7.63816833e+00, 8.80350876e+00,
                                    1.02970390e+01, 1.21807995e+01, 1.48209810e+01, 1.59163733e+01, 1.49712276e+01,
                                    1.13557968e+01, 7.22266293e+00, 4.51211548e+00]]).to(args.device)
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
                ims = ims.permute(0, 3, 1, 2)
                # Normalization
                ims /= 255
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

                loss = MSE_criterion(pred, ims[:, 5:]) + 0.57 * freq_loss1

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

                loss = MSE_criterion(pred, ims[:, 5:]) + 0.57 * freq_loss2

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
    parser.add_argument('--dataset', type=str, default='sevir', help='dataset name')
    parser.add_argument('--patch_size', type=int, default=4, help='')
    parser.add_argument('--img_width', type=int, default=128, help='image size')
    parser.add_argument('--img_channel', type=int, default=1, help='')

    # Reverse scheduled sampling
    parser.add_argument('--reverse_scheduled_sampling', type=int, default=1)
    parser.add_argument('--r_sampling_step_1', type=int, default=25000)
    parser.add_argument('--r_sampling_step_2', type=float, default=50000)
    parser.add_argument('--r_exp_alpha', type=float, default=5000)

    # Training hyperparameters
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='use cpu/gpu')
    parser.add_argument('--gpus', type=str, default='0', help='gpu device ID')
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
