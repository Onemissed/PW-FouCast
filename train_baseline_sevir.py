import os
import math
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from generator_sevir_withpangu import DataGenerator
from util.preprocess import reshape_patch
from load_earthformer_cfg import load_earthformer_config

from model.predrnnv2 import PredRNNv2_model
from model.simvpv2 import SimVP_model
from model.tau_model import TAU_model
from model.earthformer import CuboidTransformer_model
from model.pastnet import PastNet_model
from model.afno import AFNO_model
from evaluation.scores_rnn_sevir import Model_eval_RNN
from evaluation.scores_non_rnn_sevir import Model_eval_nonRNN

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
        'predrnn_v2': PredRNNv2_model,
        'simvp_v2': SimVP_model,
        'tau': TAU_model,
        'earthformer': CuboidTransformer_model,
        'pastnet': PastNet_model,
        'afno': AFNO_model,
    }
    ModelClass = MODEL_REGISTRY.get(args.model.lower())
    if ModelClass is None:
        raise ValueError(f'Unknown model: {args.model!r}. '
                         f'Available models: {list(MODEL_REGISTRY)}')

    # Load the model configuration
    if args.model == 'earthformer':
        model_kwargs = load_earthformer_config('sevir')
    else:
        model_kwargs = load_model_config(args.model.lower(), 'config/sevir')
        model_kwargs['args'] = args
    model = ModelClass(**model_kwargs).to(args.device)
    model = torch.nn.DataParallel(model)

    # Loss function
    MSE_criterion = nn.MSELoss(reduction='mean')

    # Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epoch)

    # eval
    if args.model is 'predrnn_v2':
        model_eval_testdata = Model_eval_RNN(args)
    else:
        model_eval_testdata = Model_eval_nonRNN(args)

    for epoch in range(args.epoch):
        print("epoch:", epoch + 1)
        with tqdm(total=len(train_loader)) as pbar:
            model.train()
            for i, X in enumerate(train_loader):
                # For RNN models
                if args.model is 'predrnn_v2':
                    ims = X.numpy().astype(np.float32)
                    # [B, T, H, W]
                    ims = np.transpose(ims, (0, 3, 1, 2))
                    ims = np.expand_dims(ims, axis=4)
                    # Split into patch
                    ims = reshape_patch(ims, args.patch_size)
                    # Normalization
                    ims /= 255
                    ims = torch.FloatTensor(ims).to(args.device)
                    # Reverse scheduled sampling
                    itr = epoch * len(train_loader) + i
                    real_input_flag = reserve_schedule_sampling_exp(itr, ims.shape[0], args)

                    next_frames, decouple_loss = model(ims, real_input_flag)
                    decouple_loss = torch.mean(decouple_loss)
                    loss = MSE_criterion(next_frames, ims[:, 1:]) + 0.1 * decouple_loss

                # For Non-RNN models
                else:
                    ims = X.float().to(args.device)
                    # [B, T, H, W]
                    ims = ims.permute(0, 3, 1, 2)
                    # Normalization
                    ims /= 255
                    if args.model == 'earthformer':
                        # For earthformer, change the tensor shape to [B, T, H, W, C]
                        ims = ims.unsqueeze(dim=4)
                    else:
                        # [B, T, C, H, W]
                        ims = ims.unsqueeze(dim=2)
                    pred = model(ims[:, :5])

                    loss = MSE_criterion(pred, ims[:, 5:])

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

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
                        choices=['afno', 'predrnn_v2', 'simvp_v2', 'tau', 'earthformer', 'pastnet'],
                        help='The model used for training')
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
