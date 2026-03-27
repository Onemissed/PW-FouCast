import os
import math
import yaml
import random
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from generator_sevir_withpangu import DataGenerator
from util.preprocess import reshape_patch
from load_earthformer_cfg import load_earthformer_config
from evaluation.scores_sevir import Model_eval
from safetensors.torch import load_model
from huggingface_hub import hf_hub_download

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
        'pw_foucast': PW_FouCast,
    }
    ModelClass = MODEL_REGISTRY.get(args.model.lower())
    if ModelClass is None:
        raise ValueError(f'Unknown model: {args.model!r}. '
                         f'Available models: {list(MODEL_REGISTRY)}')

    # Load the model configuration
    if args.model == 'earthformer':
        model_kwargs = load_earthformer_config(f'{args.dataset}')
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
    print("using gpu: ", args.gpus)
    model = ModelClass(**model_kwargs).to(args.device)
    model = torch.nn.DataParallel(model)

    # data
    # train_data = DataGenerator(train_list, train_list_pangu)
    # train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
    test_data = DataGenerator(test_list, test_list_pangu)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False, num_workers=0)

    # eval
    model_eval_testdata = Model_eval(args)

    epoch = 0

    # Download pretrained weights
    weights_path = hf_hub_download(repo_id=f"Onemiss/PW-FouCast", filename=f"{args.model}/{args.dataset}/model.safetensors")
    load_model(model, weights_path)

    model.eval()
    model_eval_testdata.eval(test_loader, model, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pw_foucast',
                        choices=['pw_foucast', 'afno', 'predrnn_v2', 'simvp_v2', 'tau', 'earthformer', 'pastnet', 'alphapre',
                                 'nowcastnet', 'lmc_memory', 'lightnet', 'mm_rnn', 'stjointnet'],
                        help='The model used for training')
    parser.add_argument('--dataset', type=str, default='sevir', help='dataset name')
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
    parser.add_argument('--gpus', type=str, default='1', help='gpu device ID')
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

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
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
    from model.pw_foucast import PW_FouCast

    DoTrain(args)
