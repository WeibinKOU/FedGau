from data import Dataset
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
import time

import numpy as np
import pandas as pd

import imgaug as ia
import imgaug.augmenters as iaa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)
print(torch.__version__)


'''
#4.3.1
models = ['CrackNet', 'BiSeNetV2', 'SegNet', 'DeepLabv3']
model_paths = ['checkpoints/cent_CSBiSeNetV2/epoch-500.pt',
        'checkpoints/cent_BiSeNetV2/epoch-220.pt',
        'checkpoints/cent_SegNet/epoch-500.pt',
        'checkpoints/cent_DeepLabv3/epoch-500.pt']
labels = ['CrackNet', 'BiSeNetV2', 'SegNet', 'DeepLabv3']
TESTDATA_DIR = '../datasets/'
'''
'''
#4.3.2
models = ['CrackNet', 'BiSeNetV2', 'SegNet', 'DeepLabv3']
model_paths = ['checkpoints/fed_CSBiSeNetV2_default/Cloud_FL_499_model.pth',
        'checkpoints/fed_BiSeNetV2_default_sec/Cloud_FL_179_model.pth',
        'checkpoints/fed_SegNet_default/Cloud_FL_499_model.pth',
        'checkpoints/fed_DeepLabv3_default/Cloud_FL_9_model.pth']
labels = ['FL-CrackNet', 'FL-BiSeNetV2', 'FL-SegNet', 'FL-DeepLabv3']
TESTDATA_DIR = '../datasets/'

'''
'''
#4.3.3
models = ['CrackNet', 'CrackNet', 'CrackNet', 'CrackNet', 'CrackNet']
model_paths = ['checkpoints/fed_CSBiSeNetV2_default/Cloud_FL_499_model.pth',
        'checkpoints/cent_CSBiSeNetV2/epoch-500.pt',
        'checkpoints/IL_exc_#1/Cloud_FL_499_model.pth',
        'checkpoints/IL_exc_#2/Cloud_FL_499_model.pth',
        'checkpoints/IL_exc_#6/Cloud_FL_499_model.pth',]
labels = ['FL', 'CL', 'IL-EXC-#1', 'IL-EXC-#2', 'IL-EXC-#6']
#TESTDATA_DIR = '../datasets/3_Ren/'
#TESTDATA_DIR = '../datasets/5_Yang/'
TESTDATA_DIR = '../datasets/Volker/'
'''
#4.3.4
models = ['CrackNet', 'CrackNet']
model_paths = ['checkpoints/fed_CSBiSeNetV2_default/Cloud_FL_499_model.pth',
        'checkpoints/traditional_2l_FL-sec/Edge0_499_FL_model.pth']
labels = ['Proposed CEDFL', 'Traditional FL']
TESTDATA_DIR = '../datasets/'

'''
#4.3.6
models = ['CrackNet', 'CrackNet', 'CrackNet', 'CrackNet']
model_paths = ['checkpoints/fed_CSBiSeNetV2_default/Cloud_FL_49_model.pth',
        'checkpoints/fed_CSBiSeNetV2_default/Edge0_39_FL_model.pth',
        'checkpoints/fed_CSBiSeNetV2_default/Edge1_39_FL_model.pth',
        'checkpoints/fed_CSBiSeNetV2_default/Edge2_39_FL_model.pth']
labels = ['Cloud', 'Edge1', 'Edge2', 'Edge3']
TESTDATA_DIR = '../datasets/'
'''

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Palm Detector')
    parser.add_argument('--gpu', type=int, default=0, help='choose GPU to use')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--output', type=str, default='runs/test/', help='output folder')
    parser.add_argument("--model", type=str, default='', help="To specify which model to be trained, options: [CrackNet, BiSeNetV2, SegNet, DeepLabv3]")
    parser.add_argument('--model-path', type=str, default=None, help='model path')
    parser.add_argument('--img-size', type=str, default='320,320', help='the uniform size of the resized images')
    args = parser.parse_args()
    return args

def calc_metrics(gt_mask, pred_mask):
    from sklearn.metrics import jaccard_score as jaccard
    from sklearn.metrics import precision_score as precision
    from sklearn.metrics import recall_score as recall
    from sklearn.metrics import f1_score as f1

    iou = jaccard(gt_mask.flatten(), pred_mask.flatten())
    prec = precision(gt_mask.flatten(), pred_mask.flatten())
    recall = recall(gt_mask.flatten(), pred_mask.flatten())
    f_one = f1(gt_mask.flatten(), pred_mask.flatten())

    return iou.item(), prec.item(), recall.item(), f_one.item()

def plot_pr_curve(gt, pred, comment='Default'):
    from sklearn.metrics import precision_recall_curve, auc

    precision, recall, thresholds = precision_recall_curve(gt, pred)
    auc_under_pr = auc(recall, precision)

    plt.plot(recall, precision,'-',label='%s - AUC = %0.4f' % (comment, auc_under_pr))

if __name__ == "__main__":
    args = build_parser()

    models = models if args.model == '' else [args.model]
    model_paths = model_paths if args.model == '' else [args.model_path]
    labels = labels if args.model == '' else [args.model]

    prec_rec_curve = plt.figure()

    for model, model_path, label in zip(models, model_paths, labels):
        print(model, model_path, label)
        model_sel = {}
        if model == 'CrackNet':
            model_sel['model'] = CrackNet
        elif model == 'BiSeNetV2':
            model_sel['model'] = BiSeNetV2
        elif model == 'SegNet':
            model_sel['model'] = SegNet
        elif model == 'DeepLabv3':
            model_sel['model'] = DeepLabv3

        device = torch.device('cuda:%d' % (args.gpu) if torch.cuda.is_available() else 'cpu')
        model = model_sel['model'](n_classes=2)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=str(device)))
        model.aux_mode = 'test'
        model.eval()

        new_height, new_width = [int(x) for x in args.img_size.split(',')]
        seq = iaa.Sequential([
            iaa.Resize({'height': new_height, 'width': new_width}),
        ])

        test_dataset = Dataset(TESTDATA_DIR, seq, type_='test')
        dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=1,
            shuffle=False)

        iou = []
        precision = []
        recall = []
        f1 = []
        start_time = time.time()
        pr_masks = []
        pr_pred_masks = []
        for imgs, imgs_inv, masks, names in tqdm(dataloader):
            pred_masks = F.softmax(model(imgs.to(device), imgs_inv.to(device)), dim=1)[:, 0, :, :]

            pr_m = masks
            pr_pm = pred_masks
            pr_masks += list(pr_m.squeeze().numpy().astype(int).reshape(1, -1).squeeze())
            pr_pred_masks += list(pr_pm.squeeze().detach().cpu().numpy().reshape(1, -1).squeeze())

            t = pred_masks
            t[t >= 0.3] = 1.0
            t[t < 0.3] = 0.0
            pred_masks = t
            masks = masks.squeeze().numpy().astype(int)
            pred_masks = pred_masks.squeeze().detach().cpu().numpy().astype(int)
            for i in range(masks.shape[0]):
                pm = pred_masks[i] * 255
                cv2.imwrite('./test/' + names[i], pm)
                iou_t, pre_t, rec_t, f1_t = calc_metrics(masks[i], pred_masks[i])
                iou.append(iou_t)
                precision.append(pre_t)
                recall.append(rec_t)
                f1.append(f1_t)
        end_time = time.time()
        diff_time = end_time - start_time
        print('IOU max and avg: ', max(iou), sum(iou) / len(iou))
        print('precision max and avg: ', max(precision), sum(precision) / len(precision))
        print('recall max and avg: ', max(recall), sum(recall) / len(recall))
        print('f1 max and avg: ', max(f1), sum(f1) / len(f1))
        print('inference time: ', diff_time / len(iou))

        plot_pr_curve(pr_masks, pr_pred_masks, label)

    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.savefig("Precision_recall.png")
