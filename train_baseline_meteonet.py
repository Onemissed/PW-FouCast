import os
import math
import yaml
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from generator_meteonet_withpangu import DataGenerator
from util.preprocess import reshape_patch
from load_earthformer_cfg import load_earthformer_config
from evaluation.scores_sevir import Model_eval

def schedule_sampling(eta, itr, args, batchsize):
    zeros = np.zeros((batchsize,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (batchsize, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(batchsize):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (batchsize,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def reserve_schedule_sampling_exp(itr, batch_size, args):
    T, img_channel, img_height, img_width = args.in_shape
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample((batch_size, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample((batch_size, args.output_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((img_height // args.patch_size,
                    img_width // args.patch_size,
                    args.patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height // args.patch_size,
                      img_width // args.patch_size,
                      args.patch_size ** 2 * img_channel))

    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - 2,
                                  img_height // args.patch_size,
                                  img_width // args.patch_size,
                                  args.patch_size ** 2 * img_channel))
    return torch.FloatTensor(real_input_flag).to(args.device)

def load_config(model_name, cfg_root='config/sevir'):
    path = os.path.join(cfg_root, f'{model_name.lower()}.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config for model '{model_name}' not found at {path!r}")

    with open(path, 'r') as f:
        return yaml.safe_load(f)

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

    MODEL_REGISTRY = {
        'predrnn_v2': PredRNNv2_model,
        'simvp_v2': SimVP_model,
        'tau': TAU_model,
        'earthformer': CuboidTransformer_model,
        'pastnet': PastNet_model,
        'alphapre': get_model,
        'nowcastnet': Net,
        'lmc_memory': Predictor,
        'afno': AFNO_model,
        'lightnet': LightNet_Model,
        'mm_rnn': Model,
        'stjointnet': CM_STJointNet,
    }
    ModelClass = MODEL_REGISTRY.get(args.model.lower())
    if ModelClass is None:
        raise ValueError(f'Unknown model: {args.model!r}. '
                         f'Available models: {list(MODEL_REGISTRY)}')

    # Load the model configuration
    if args.model == 'earthformer':
        model_kwargs = load_earthformer_config('meteonet')
    else:
        config = load_config(args.model.lower())
        model_kwargs = config.get('model', {})
        model_kwargs['args'] = args
        train_cfg = config.get('train', {})
        batch_size = train_cfg.get('batch_size', 16)  # 16 is a fallback default
        model_lr = train_cfg.get('learning_rate', 1e-3)
        args.batchsize = batch_size
        args.lr = model_lr
    print("dataset: ", args.dataset)
    print("model: ", args.model)
    print("lr: ", args.lr)
    print("batch size: ", args.batchsize)
    model = ModelClass(**model_kwargs).to(args.device)
    model = torch.nn.DataParallel(model)

    # data
    train_data = DataGenerator(train_list)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
    test_data = DataGenerator(test_list)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False, num_workers=0)

    # Loss function
    MSE_criterion = nn.MSELoss(reduction='mean')

    # Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.model == 'lmc_memory':
        total_steps = 2 * len(train_loader) * args.epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epoch)

    # eval
    model_eval_testdata = Model_eval(args)

    eta = args.sampling_start_value

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
                ims_pangu = X_pangu.float().to(args.device)
                B, T, var, pressure_level, H, W = ims_pangu.shape
                # Normalize Pangu-Weather forecasts
                ims_pangu = (ims_pangu - mean_pangu) / std_pangu
                ims_pangu = ims_pangu[:, :, :, 2:6, :, :]
                ims_pangu = ims_pangu.reshape(B, T, -1, H, W)

                # For RNN models
                if args.model in ['predrnn_v2', 'mm_rnn']:
                    ims = X.numpy().astype(np.float32)
                    # [B, T, H, W]
                    ims = np.expand_dims(ims, axis=4)
                    # Split into patch
                    ims = reshape_patch(ims, args.patch_size)
                    # Normalization
                    ims /= 80
                    ims = torch.FloatTensor(ims).to(args.device)
                    itr = epoch * len(train_loader) + i
                    # Reverse scheduled sampling
                    if args.model == 'predrnn_v2':
                        real_input_flag = reserve_schedule_sampling_exp(itr, ims.shape[0], args)
                        next_frames, decouple_loss = model(ims, real_input_flag)
                        decouple_loss = torch.mean(decouple_loss)
                        loss = MSE_criterion(next_frames, ims[:, 1:]) + 0.1 * decouple_loss
                    else:
                        eta, real_input_flag = schedule_sampling(eta, itr, args, B)
                        real_input_flag = torch.FloatTensor(real_input_flag).to(args.device)
                        next_frames = model(ims, ims_pangu, real_input_flag)
                        loss = MSE_criterion(next_frames, ims[:, 5:])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                elif args.model == 'lmc_memory':
                    ims = X.float().to(args.device)
                    # Normalization
                    ims /= 80
                    ims = ims.unsqueeze(dim=2)

                    # define data indexes
                    short_start, short_end = 0, args.short_len
                    long_start = np.random.randint(0, args.short_len + args.out_len - args.long_len + 1)
                    long_end = long_start + args.long_len
                    short_data = ims[:, short_start:short_end, :, :, :]
                    long_data = ims[:, long_start:long_end, :, :, :]

                    # Phase 1
                    model.module.memory.memory_w.requires_grad = True  # train memory weights
                    pred = model(short_data, long_data, args.out_len, phase=1)
                    loss = MSE_criterion(pred, ims[:, 5:])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Phase 2
                    model.module.memory.memory_w.requires_grad = False  # do not train memory weights
                    pred = model(short_data, None, args.out_len, phase=2)
                    loss = MSE_criterion(pred, ims[:, 5:])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # For Non-RNN models
                else:
                    ims = X.float().to(args.device)
                    # Normalization
                    ims /= 80
                    if args.model in ['earthformer', 'nowcastnet']:
                        ims = ims.unsqueeze(dim=4)
                    else:
                        # [B, T, C, H, W]
                        ims = ims.unsqueeze(dim=2)

                    if args.model in ['earthformer', 'nowcastnet', 'afno']:
                        pred = model(ims[:, :5])
                    elif args.model in ['simvp_v2', 'tau', 'pastnet']:
                        pred_1 = model(ims[:, :5])
                        pred_2 = model(pred_1)
                        pred_3 = model(pred_2)
                        pred_4 = model(pred_3)
                        pred = torch.cat((pred_1, pred_2, pred_3, pred_4), dim=1)
                    elif args.model == 'alphapre':
                        _, loss_dict = model.module.predict(frames_in=ims[:, :5], frames_gt=ims[:, 5:], compute_loss=True)
                    elif args.model in ['lightnet', 'stjointnet']:
                        pred = model(ims[:, :5], ims_pangu)

                    if args.model == 'alphapre':
                        loss = loss_dict['total_loss']
                    else:
                        loss = MSE_criterion(pred, ims[:, 5:])
                    optimizer.zero_grad()
                    loss.backward()
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
    parser.add_argument('--model', type=str, default='afno',
                        choices=['afno', 'predrnn_v2', 'simvp_v2', 'tau', 'earthformer', 'pastnet', 'alphapre',
                                 'nowcastnet', 'lmc_memory', 'lightnet', 'mm_rnn', 'stjointnet'],
                        help='The model used for training')
    parser.add_argument('--dataset', type=str, default='meteonet', help='dataset name')
    parser.add_argument('--filter_size', type=int, default=5, help='kernel size')
    parser.add_argument('--stride', type=int, default=1, help='')
    parser.add_argument('--layer_norm', type=int, default=1, help='whether use layer norm in RNN')
    parser.add_argument('--patch_size', type=int, default=4, help='')
    parser.add_argument('--img_width', type=int, default=128, help='image size')
    parser.add_argument('--img_channel', type=int, default=1, help='')
    parser.add_argument('--short_len', type=int, default=5, help='number of short-term frames')
    parser.add_argument('--long_len', type=int, default=20, help='number of long-term frames')
    parser.add_argument('--out_len', type=int, default=20, help='number of predicted frames')

    # Reverse scheduled sampling
    parser.add_argument('--r_sampling_step_1', type=int, default=25000)
    parser.add_argument('--r_sampling_step_2', type=float, default=50000)
    parser.add_argument('--r_exp_alpha', type=float, default=5000)

    # Scheduled sampling
    parser.add_argument('--sampling_stop_iter', type=int, default=50000)
    parser.add_argument('--sampling_start_value', type=float, default=1.0)
    parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

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

    from model.predrnnv2 import PredRNNv2_model
    from model.simvpv2 import SimVP_model
    from model.tau_model import TAU_model
    from model.earthformer import CuboidTransformer_model
    from model.pastnet import PastNet_model
    from model.alphapre import get_model
    from model.nowcastnet import Net
    from model.lmc_memory import Predictor
    from model.afno import AFNO_model
    from model.lightnet import LightNet_Model
    from model.mm_rnn import Model
    from model.stjointnet import CM_STJointNet

    DoTrain(args)
