import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label, depth):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
            depth = F.pad(depth, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
            depth = F.pad(depth, (self.size[1] - depth.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            depth = F.pad(depth, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w), F.crop(depth, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label, depth):
        for t in self.transforms:
            image, label, depth = t(image, label, depth)
        return image, label, depth


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label, depth):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label), F.hflip(depth)
        return img, label, depth


class PairRandomVerticalFilp(transforms.RandomVerticalFlip):
    def __call__(self, img, label, depth):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label), F.vflip(depth)
        return img, label, depth


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label, depth):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label), torch.tensor(np.array(depth)).unsqueeze(0)    # F.to_tensor(depth)
