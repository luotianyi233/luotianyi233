import os
import torch
import numpy as np
from PIL import Image as Image
from data import *
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
# TODO: 放到main.py
depth_scale, depth_max = 1000, 65.535       # 65.535来自tools/find_depth_max.py(最大深度距离为65.535米)


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose([
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairRandomVerticalFilp(),
                PairToTensor()
        ] )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # prefetch_factor=batch_size
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=batch_size,
        pin_memory=True
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        import glob
        self.image_list_blur = glob.glob(os.path.join(self.image_dir, "*/blur/*.png"))
        self.image_list_sharp = glob.glob(os.path.join(self.image_dir, "*/clear/*.png"))
        self.image_list_depth = glob.glob(os.path.join(self.image_dir, "*/depth/*.png"))
        assert len(self.image_list_blur) == len(self.image_list_sharp) == len(self.image_list_depth)
        # self._check_image(self.image_list_blur)
        # self._check_image(self.image_list_sharp)
        # self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.depth_scale, self.depth_max = depth_scale, depth_max

    def __len__(self):
        return len(self.image_list_blur)

    def __getitem__(self, idx):
        image = Image.open(self.image_list_blur[idx])
        label = Image.open(self.image_list_sharp[idx])
        depth = Image.open(self.image_list_depth[idx])

        if self.transform:
            image, label, depth = self.transform(image, label, depth)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            depth = torch.tensor(np.array(depth)).unsqueeze(0)      # F.to_tensor(depth)
        depth = depth.float()/self.depth_scale/self.depth_max       # normalize depth
        if self.is_test:
            name = self.image_list_blur[idx]
            return image, label, depth, name
        return image, label, depth

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
