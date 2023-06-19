from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from utils.utils_cla import progress_bar
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

from model.fasterrcnn.frcnn_training import FasterRCNNTrainer
import os
from utils.utils import get_lr
from utils.utils_map import get_map
from utils.callbacks import EvalCallback
from utils.utils import get_classes

def SS_calc_metric(gt_mask, pred_mask):
    t = pred_mask
    t[t >= 0.5] = 1
    t[t < 0.5] = 0
    pred_mask = t

    gt_mask[gt_mask >= 0.5] = 1
    gt_mask[gt_mask < 0.5] = 0

    gt_mask = gt_mask.astype(int).flatten()
    pred_mask = pred_mask.astype(int).flatten()

    intersection = np.sum(gt_mask * pred_mask)

    iou = intersection / (np.sum(gt_mask + pred_mask <= 2) + 1e-6)
    prec = intersection / (np.sum(pred_mask == 1) + 1e-6)
    reca = intersection / (np.sum(gt_mask == 1) + 1e-6)
    f_one = 2 * prec * reca / (prec + reca + 1e-6)

    return iou, prec, reca, f_one

def SS_Evaluate(model, dataloader, dev):
    model.eval()
    model.aux_mode = 'test'
    cla_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafficlight', 'trafficsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    with torch.no_grad():
        iou = []
        precision = []
        recall = []
        f1 = []
        res_dict = {}
        for imgs, masks, names in tqdm(dataloader):
            pred_masks = F.softmax(model(imgs.to(dev)), dim=1)
            masks = masks.squeeze().numpy()
            pred_masks = pred_masks.squeeze().detach().cpu().numpy()

            for j in range(masks.shape[1]):
                cla_dict = {}
                cla_iou = []
                cla_precision = []
                cla_recall = []
                cla_f1 = []
                for i in range(masks.shape[0]):
                    iou_t, pre_t, rec_t, f1_t = SS_calc_metric(masks[i,j], pred_masks[i,j])
                    cla_iou.append(iou_t)
                    cla_precision.append(pre_t)
                    cla_recall.append(rec_t)
                    cla_f1.append(f1_t)
                cla_avg_iou = sum(cla_iou) / len(cla_iou)
                cla_avg_pre = sum(cla_precision) / len(cla_precision)
                cla_avg_recall = sum(cla_recall) / len(cla_recall)
                cla_avg_f1 = sum(cla_f1) / len(cla_f1)

                cla_dict['IoU'] = cla_avg_iou
                cla_dict['Precision'] = cla_avg_pre
                cla_dict['Recall'] = cla_avg_recall
                cla_dict['F1'] = cla_avg_f1
                res_dict[cla_names[j]] = cla_dict

            iou.append(cla_avg_iou)
            precision.append(cla_avg_pre)
            recall.append(cla_avg_recall)
            f1.append(cla_avg_f1)

        avg_iou = sum(iou) / len(iou)
        avg_pre = sum(precision) / len(precision)
        avg_recall = sum(recall) / len(recall)
        avg_f1 = sum(f1) / len(f1)

        res_dict['mIoU'] = avg_iou
        res_dict['mPrecision'] = avg_pre
        res_dict['mRecall'] = avg_recall
        res_dict['mF1'] = avg_f1
    model.train()
    return res_dict

def classi_Evaluate(model, testloader, dev):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(dev), targets.to(dev)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    model.train()

    return test_loss / len(testloader), 100.*correct/total


def objDect_Evaluate(model, testloader, dev, val_lines, epoch_step_val, logdir, classes_path, epoch, input_shape):
    val_loss = 0
    optimizer = optim.Adam(model.parameters())
    train_util = FasterRCNNTrainer(model, optimizer)


    class_names, num_classes = get_classes(classes_path)
    eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, logdir, torch.cuda.is_available())
    

    with tqdm(total=epoch_step_val, postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(testloader):
            if iteration >= epoch_step_val:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                images = images.to(dev)

                train_util.optimizer.zero_grad()
                _, _, _, _, val_total = train_util.forward(images, boxes, labels, 1)
                val_loss += val_total.item()
                
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1)})
                pbar.update(1)

    ap_dict = eval_callback.on_epoch_end(epoch)
    
    return val_loss / epoch_step_val, ap_dict
