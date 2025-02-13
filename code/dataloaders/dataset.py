import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f_image = Image.open(self._base_dir + "/tissue-train-100/images_size_256/{}.jpg".format(case))
            h5f_label = Image.open(self._base_dir + "/tissue-train-100/label_size_256/{}.jpg".format(case))
        elif self.split == "val":
            h5f_image = Image.open(self._base_dir + "/val/images/{}.png".format(case))
            h5f_label = Image.open(self._base_dir + "/val/masks/{}.png".format(case))
        else:
            h5f_image = Image.open(self._base_dir + "/tissue-test/images/{}.jpg".format(case))
            h5f_label = Image.open(self._base_dir + "/tissue-test/labels/{}.png".format(case))
            print(self._base_dir + "/tissue-test/labels/{}.png".format(case))
        #image = h5f["image"][:]
        #label = h5f["label"][:]
        h5f_label = h5f_label.convert("L")
        h5f_image=np.array(h5f_image)
        h5f_label=np.array(h5f_label)
        h5f_image = np.transpose(h5f_image, (2, 0, 1))
        
        sample = {"image": h5f_image, "label": h5f_label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k,axes=(1, 2))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def  random_affine(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
        image = np.transpose(image, (1, 0, 2))
    degrees = 30  # 旋转的最大角度范围
    translate = (0.2, 0.2)  # 平移的最大比例范围
    scale = (0.8, 1.2)  # 缩放的范围
    shear = 10  # 剪切的最大角度范围

    # 创建随机仿射变换对象
    random_affine = transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear)

    # 应用随机仿射变换
    transformed_image = random_affine(image)

    # 将变换后的图像转回 NumPy 数组，如果需要
    #transformed_image_np = np.array(transformed_image)
    return transformed_image

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def random_affine_with_cutout(image, cutout_prob=0.5):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
        image = np.transpose(image, (1, 0, 2))

    # 随机仿射变换参数
    degrees = 30
    translate = (0.2, 0.2)
    scale = (0.8, 1.2)
    shear = 10

    # 创建随机仿射变换对象
    random_affine = transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear)

    # 应用随机仿射变换
    transformed_image = random_affine(image)

    # 使用RandomErasing添加Cutout
    if np.random.rand() < cutout_prob:
        erasing_prob = 0.5  # 擦除的概率
        max_area = 0.2  # 擦除区域的最大比例

        random_erasing = transforms.RandomErasing(p=erasing_prob, scale=(0.02, max_area))
        transformed_image = random_erasing(image)

    # 将变换后的图像转回 NumPy 数组，如果需要
    # transformed_image_np = np.array(transformed_image)

    return transformed_image

def add_random_noise(image, noise_factor=0.1):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
        image = np.transpose(image, (1, 0, 2))

    # 创建颜色抖动对象，通过调整亮度、对比度、饱和度和色相来引入随机噪声
    color_jitter = transforms.ColorJitter(brightness=noise_factor,
                                          contrast=noise_factor,
                                          saturation=noise_factor,
                                          hue=noise_factor)

    # 应用颜色抖动
    noisy_image = color_jitter(image)

    # 将变换后的图像转回 NumPy 数组，如果需要
    # noisy_image_np = np.array(noisy_image)

    return noisy_image


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
        image = np.transpose(image, (1, 0, 2))

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        x, y ,z= image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y,3), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y,3), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        
        
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        #image_weak, label = random_rotate(image, label)
        #print("image_weak.shape",image_weak.shape)
        # strong augmentation is color jitter
        #image_strong=image_weak
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        #image_strong = add_random_noise(image_weak).type("torch.FloatTensor")
        
        #image_strong = random_affine_with_cutout(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image,
            "image_strong": image,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        #print("image.shape",image.shape)
        #_,x, y = image.shape
        #print(x,y)
        
        return zoom(image, (self.output_size[0] / image.shape[0], self.output_size[1] / image.shape[1], self.output_size[2] / image.shape[2]), order=0)
    def resize_label(self,image):
        return zoom(image, (self.output_size[1] / image.shape[0], self.output_size[2] / image.shape[1]), order=0)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

