import os
import torch
import argparse
from torch.backends import cudnn
from models.SFNet import build_net
from train import _train
from eval import _eval

def main(args):
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    mode = [args.mode, args.data]
    model = build_net(mode)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    if args.mode == 'test':
        _eval(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='SFNet', type=str)

    parser.add_argument('--data_dir', type=str, default='GOPRO')

    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--data', type=str, default='GOPRO', choices=['GOPRO', 'HIDE'])

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')

    # Test
    parser.add_argument('--test_model', type=str, default='GOPRO.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, args.data, 'Training-Results/')
    args.result_dir = os.path.join('results/', args.model_name, args.data)


    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/SFNet.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)

    print(args)
    main(args)
