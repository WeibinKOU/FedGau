from data import Dataset
from utils.average_meter import AverageMeter
from utils.logger import Logger
from utils.func import *
from config import *

from model.bisenetv2 import BiSeNetV2
from model.cracknet import CrackNet
from model.segnet import SegNet
from model.deeplabv3 import DeepLabv3

import os
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import imgaug as ia
import imgaug.augmenters as iaa
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from utils.util_semseg import SS_Evaluate

np.random.seed(0)
torch.manual_seed(0)
print(torch.__version__)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs for training (default: 3000)')
    parser.add_argument('--gpu', type=int, default=0, help='choose which GPU to use')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--img-size', type=str, default='320,320', help='the uniform size of the resized images')
    parser.add_argument('--save-epoch', type=int, default=10, metavar='N',
                        help='num of epochs saving the model (default: 10)')
    parser.add_argument("--model", type=str, default='DeepLabv3', help="To specify which model to be trained, options: [CrackNet, BiSeNetV2, SegNet, DeepLabv3]")
    parser.add_argument("--dataset-path", type=str, default='./datasets/cityspace/centralized/', help="To specify training dataset path")
    args = parser.parse_args()
    return args

def save_cmd(log_dir):
    txt_file = open(log_dir +'/cmd.txt', 'w')
    cmd=" ".join("\"" + arg + "\"" if " " in arg else arg for arg in sys.argv)
    cmd = 'python ' + cmd
    txt_file.write(cmd)
    txt_file.close()

if __name__ == "__main__":
    args = build_parser()
    tb = SummaryWriter('./logs/centralized/deeplabv3_cityspace/')
    save_cmd(tb.logdir)
    log_name = tb.logdir.split('/')[-1]
    logger = Logger("train-%s" % (log_name), tb.logdir)

    model_sel = {}
    if args.model == 'CrackNet':
        model_sel['model'] = CrackNet
    elif args.model == 'BiSeNetV2':
        model_sel['model'] = BiSeNetV2
    elif args.model == 'SegNet':
        model_sel['model'] = SegNet
    elif args.model == 'DeepLabv3':
        model_sel['model'] = DeepLabv3

    #new_height, new_width = [int(x) for x in args.img_size.split(',')]
    #seq = iaa.Sequential([
        #iaa.Fliplr(.5),
        #iaa.Affine(
            #scale=(.75, 1.33),
            #translate_percent={'x': (-.05, .05), 'y': (-.05, .05)},
            #rotate=(-25, 25)
        #),
        #iaa.GammaContrast((.4, 2.5)),
        #iaa.GaussianBlur((0, 3.0)),

        # Resize images to given size
        #iaa.Resize({'height': new_height, 'width': new_width}),
    #])

    train_dataset = Dataset(args.dataset_path)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=True)

    test_dataset = Dataset('./datasets/cityspace/test/', type_='test')
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=1)

    model_path = get_dir(tb.logdir, log_name)

    device = torch.device('cuda:%d' % (args.gpu) if torch.cuda.is_available() else 'cpu')
    model = model_sel['model'](n_classes=19)
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.aux_mode = 'train'
        loss = AverageMeter()
        for imgs, masks, names in tqdm(dataloader):
            #*logits, = model(imgs.to(device), imgs_inv.to(device))
            *logits, = model(imgs.to(device))
            pred_masks = [F.softmax(logit, dim=1) for logit in logits]
            dist = [criterion(pred_mask, masks.to(device)) for pred_mask in pred_masks]
            dist = sum(dist)
            optimizer.zero_grad()
            dist.backward()
            loss.update(dist.item())
            optimizer.step()
        logger.info("Train epoch: %2d, loss=%.4f" % (epoch, loss.avg))
        tb.add_scalar('Train.Loss', loss.avg, epoch)

        if epoch % 5 != 4:
            continue

        dicts= SS_Evaluate(model, test_dataloader, device)
        for k, v in dicts.items():
            if k == 'mIoU':
                tb.add_scalar('Eval.mIOU', v, epoch)
            elif k == 'mPrecision':
                tb.add_scalar('Eval.mPrecision', v, epoch)
            elif k == 'mRecall':
                tb.add_scalar('Eval.mRecall', v, epoch)
            elif k == 'mF1':
                tb.add_scalar('Eval.mF1', v, epoch)
            else:
                tb.add_scalar('Eval.' + k + '.IoU', v['IoU'], epoch)
                tb.add_scalar('Eval.' + k + '.Precision', v['Precision'], epoch)
                tb.add_scalar('Eval.' + k + '.Recall', v['Recall'], epoch)
                tb.add_scalar('Eval.' + k + '.F1', v['F1'], epoch)

        log_info = "[Cloud FL: %d] [Eval.AvgIOU: %f, Eval.AvgPrecision: %f, Eval.AvgRecall: %f, Eval.AvgF1: %f]" % (epoch, dicts['mIoU'], dicts['mPrecision'], dicts['mRecall'], dicts['mF1'])
        print(log_info)

        if epoch % 50 == 49:
            save_path = os.path.join(model_path, "epoch-%d.pt" % epoch)
            torch.save(model.state_dict(), save_path)
    tb.close()
