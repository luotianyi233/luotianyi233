import os
import torch
import argparse
from torch.backends import cudnn
import models
from models.DepthDeblurDAVANet import DepthDeblurDAVANet
from train import _train
from eval import _eval, _eval_self_ensemable
from datetime import datetime as dt
from utils import print_network


def main(args):
    # CUDNN
    cudnn.benchmark = True

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.model_save_dir)
    # if not os.path.exists(args.output_dir + args.model_name + '/'):
    #     os.makedirs(args.output_dir + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = models.__dict__[args.model_name].__dict__[args.model_name]()
    # print(model)
    print_network(model)

    if args.mode == 'train':
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        _train(model, args)

    elif args.mode == 'test':
        if torch.cuda.is_available():
            model.cuda()
        _eval(model, args)

    elif args.mode == 'test_self_ensemable':
        if torch.cuda.is_available():
            model.cuda()
        _eval_self_ensemable(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='DepthDeblurDAVANet', choices=['DepthDeblurDAVANet'], type=str)
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--gpu', default='0', type=str)
    # Train
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--lr_type', default='CosineAnnealingWarmRestarts',
                        choices=['CosineAnnealingWarmRestarts', 'MultiStepLR'], type=str)
    parser.add_argument('--lr_param', type=list)

    # Test
    parser.add_argument('--test_model', type=str, default='weights/')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
    args = parser.parse_args()

    args.mode = 'train'  # test,train, test_self_ensemable
    args.model_name = 'DepthDeblurDAVANet'   # DepthDeblurDAVANet
    args.data_dir = './datasets/depth_blur_dataset_separate_left_half'
    args.gpu = '2, 3'
    args.valid_freq = 5
    args.train_batch_size = 16
    args.test_batch_size = 4
    args.tag = 'origin_half-data'
    foldname = args.model_name + '_' + args.tag + '_' + dt.now().strftime("%m-%d-%H-%M-%S")
    args.lr_type = "CosineAnnealingWarmRestarts"
    args.lr_param = [20, 2]                                             # CosineAnnealingWarmRestarts对应参数
    # args.lr_param = [[(x+1) * 500 for x in range(3000//500)], 0.5]      # MultiStepLR对应参数
    # for test
    args.save_image = False
    args.test_model = "./models/"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(foldname)
    args.model_save_dir = os.path.join(args.output_dir, foldname, 'weights/')
    args.result_dir = os.path.join(args.output_dir, foldname, 'result_image/')
    args.writer_path = os.path.join(args.output_dir, foldname)
    print(args)
    main(args)
