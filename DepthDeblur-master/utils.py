import torch
import time
import numpy as np


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


def complex_depart(input):
    '''
    :param input: torch.tensor
    :return: size [*, 2]
    '''
    return torch.stack((input.real, input.imag), -1)


def PSNR(img1, img2, max=1.0):
        mse = torch.mean((img1 - img2) ** 2, (1, 2, 3))
        return (20 * torch.log10(max / torch.sqrt(mse))).mean().item()


def self_ensemable(img, rote=True, flip=True):
    imgs = [img]
    if flip:
        imgs.append(torch.flip(img, [3]))    # dim can be 2 or 3
    if rote:    # rote must behind flip
        for i in range(len(imgs)):
            imgs.append(torch.rot90(imgs[i], 1, [2, 3]))
            imgs.append(torch.rot90(imgs[i], 2, [2, 3]))
            imgs.append(torch.rot90(imgs[i], 3, [2, 3]))
    imgs.reverse()  # make the origin image be the last one
    return imgs


def merge_multi_results(imgs, rote=True, flip=True, return_avg=True):
    num = len(imgs)
    imgs = list(imgs)
    if rote:
        assert num in [4, 8]
        temp = num // 4  # if flip=True, temp=2; else tmep=1
        for i in range(num // 4):
            imgs[i * (temp + 1) + 0] = torch.rot90(imgs[i * (temp + 1) + 0], 1, [2, 3])
            imgs[i * (temp + 1) + 1] = torch.rot90(imgs[i * (temp + 1) + 1], 2, [2, 3])
            imgs[i * (temp + 1) + 2] = torch.rot90(imgs[i * (temp + 1) + 2], 3, [2, 3])
    if flip:
        assert num in [2, 8]
        for i in [0, 1, 2, 6]:
            imgs[i] = torch.flip(imgs[i], [3])
    if return_avg:
        return torch.stack(imgs, dim=-1).mean(dim=-1)
    else:
        return imgs