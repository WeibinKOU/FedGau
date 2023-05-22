import os
from time import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import transforms


class Dataset(data.Dataset):

    def __init__(self, data_dir, transform=None, type_='train'): #type: [train, test]
        super().__init__()

        self.data_dir = data_dir
        self.img_dir = self.data_dir + type_ + '/images/'
        self.mask_dir = self.data_dir + type_ + '/masks/'

        self.transform = transform
        self.img_to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        self.type_ = type_

        self.images = sorted(os.listdir(self.img_dir))
        #self.masks = sorted(os.listdir(self.mask_dir))

        self.size = len(self.images)

        if self.type_ == 'train':
            self.mask_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        name = self.images[index]
        #mask_name = self.masks[index]

        img = cv2.imread(os.path.join(self.img_dir, name), cv2.IMREAD_COLOR)[..., ::-1]
        mask = np.expand_dims(cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE), axis=2)

        img_inv = 255 * np.ones_like(img)
        img_inv = img_inv - img

        comb = np.concatenate((img, img_inv, mask), axis=2)
        new_comb = self.transform(image=comb)

        img_tensor = self.img_to_tensor(new_comb[:,:,:3])
        img_inv_tensor = self.img_to_tensor(new_comb[:,:,3:6])

        if self.type_ == 'train':
            mask_tensor_0 = torch.Tensor(new_comb[:,:,-1][np.newaxis,...] / 255.0)
            mask_tensor_1 = 1.0 - mask_tensor_0

            mask_tensor = torch.cat((mask_tensor_0, mask_tensor_1), dim=0)
            return img_tensor, img_inv_tensor, mask_tensor
        elif self.type_ == 'test':
            mask_tensor = torch.Tensor(new_comb[:,:,-1][np.newaxis,...] / 255.0)
            #mask_tensor = torch.Tensor(mask[np.newaxis,...] / 255.0)

            return img_tensor, img_inv_tensor, mask_tensor, name
