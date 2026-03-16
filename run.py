import argparse
import random

import numpy as np
import torch
import torch.backends

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.print_args import print_args


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection')

    parser.add_argument('--task_name', type=str, default='anomaly_detection',
                        help='only supports anomaly_detection')
    parser.add_argument('--is_training', type=int, default=1, help='1 for train+test, 0 for test only')
    parser.add_argument('--model_id', type=str, default='SMAP', help='experiment id')
    parser.add_argument('--model', type=str, default='AnomalyTransformer', help='model name')

    parser.add_argument('--data', type=str, default='SMAP', help='dataset: [PSM, MSL, SMAP, SMD, SWAT]')
    parser.add_argument('--root_path', type=str, default='./dataset/SMAP', help='dataset root path')
    parser.add_argument('--features', type=str, default='M', help='for compatibility')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint dir')

    parser.add_argument('--seq_len', type=int, default=100, help='window size')
    parser.add_argument('--pred_len', type=int, default=0, help='for compatibility')
    parser.add_argument('--anomaly_ratio', type=float, default=1, help='prior anomaly ratio (%%)')

    parser.add_argument('--enc_in', type=int, default=25, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=25, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='model width')
    parser.add_argument('--n_heads', type=int, default=8, help='num heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='ffn dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--itr', type=int, default=1, help='experiment repeats')
    parser.add_argument('--train_epochs', type=int, default=10, help='epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='lr scheduler type')
    parser.add_argument('--des', type=str, default='test', help='description')

    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multi gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='multi-gpu ids')

    args = parser.parse_args()
    if args.task_name != 'anomaly_detection':
        raise ValueError('Only anomaly_detection is supported in this simplified repository.')
    return args


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = get_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp_Anomaly_Detection(args)
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_dm{args.d_model}_{args.des}_{ii}'
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
    else:
        exp = Exp_Anomaly_Detection(args)
        setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_dm{args.d_model}_{args.des}_0'
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)

    if args.use_gpu:
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
