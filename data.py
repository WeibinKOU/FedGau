import os
from time import time

import cv2
import numpy as np
import pandas as pd
import pickle
import torch

from PIL import Image
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from torchvision import transforms
from utils.utils import cvtColor, preprocess_input
from MultiFL.fed_config import WIDTH, HEIGHT


class Dataset(Dataset):

    def __init__(self, data_dir, num_classes=30, type_='train'): #type: [train, test]
        super().__init__()

        self.data_dir = data_dir
        #self.img_dir = self.data_dir + type_ + '/images/'
        #self.mask_dir = self.data_dir + type_ + '/masks/'
        self.img_dir = self.data_dir + 'images/'
        self.mask_dir = self.data_dir + 'masks/'
        self.num_classes = num_classes

        self.transform = iaa.Sequential([
            iaa.Fliplr(.5),
            iaa.Affine(
                translate_percent={'x': (-.05, .05), 'y': (-.05, .05)},
                rotate=(-25, 25)
            ),

            #iaa.GammaContrast((.4, 2.5)),
            #iaa.GaussianBlur((0, 3.0)),

            iaa.Resize({'height': HEIGHT, 'width': WIDTH}),
        ])

        self.img_to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        self.type_ = type_

        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

        self.size = len(self.images)

        if self.type_ == 'train':
            self.mask_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])

    def mask_to_layers(self, mask, num_classes):
        layers = []
        for i in range(num_classes):
            layer = np.zeros_like(mask)
            layer[mask == i] = 1.0
            layers.append(layer)
        return layers

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        name = self.images[index]
        mask_name = self.masks[index]

        img = cv2.imread(os.path.join(self.img_dir, name), cv2.IMREAD_COLOR)[..., ::-1]
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask_layers = self.mask_to_layers(mask, self.num_classes)

        img_tensor = self.img_to_tensor(img)
        mask_tensor = torch.Tensor(mask_layers)

        return img_tensor, mask_tensor, name

class CIFAR10Dataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif mode == 'test':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            print("Not supported action to load dataset cifar10")
            exit()
        self.data, self.labels = self.load_single_batch()

    def load_single_batch(self):
        with open(self.data, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')

        data = batch[b'data']
        labels = np.array(batch[b'labels'])

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].reshape(3, 32, 32).astype(np.float32)
        label = self.labels[index]
        #label = np.eye(10)[label]

        sample = torch.from_numpy(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape = [600, 600], train = True):
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, y    = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data    = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box         = box_data[:, :4]
        label       = box_data[:, -1]
        return image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]

        return image_data, box

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = torch.from_numpy(np.array(images))
    return images, bboxes, labels
