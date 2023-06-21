from __future__ import print_function, absolute_import, division
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
import copy
import math

from model.fasterrcnn.frcnn_training import FasterRCNNTrainer
import os
from utils.utils import get_lr
from utils.utils_map import get_map
from utils.callbacks import EvalCallback
from utils.utils import get_classes
from collections import namedtuple

import os, sys
import platform
import fnmatch

try:
    from itertools import izip
except ImportError:
    izip = zip

CSUPPORT = True
if CSUPPORT:
    try:
        from utils.csEvaluation import addToConfusionMatrix
    except:
        CSUPPORT = False

def calculate_recall(gt_mask, pred_mask):
    true_positive = np.sum(np.logical_and(gt_mask, pred_mask))
    false_negative = np.sum(np.logical_and(gt_mask, np.logical_not(pred_mask)))
    recall = true_positive / (true_positive + false_negative + 1e-6)
    return recall

def calculate_iou(gt_mask, pred_mask):
    intersection = np.sum(np.logical_and(gt_mask, pred_mask))
    union = np.sum(np.logical_or(gt_mask, pred_mask)) + 1e-6
    iou = intersection / union
    return iou

def calculate_precision(gt_mask, pred_mask):
    true_positive = np.sum(np.logical_and(gt_mask, pred_mask))
    false_positive = np.sum(np.logical_and(np.logical_not(gt_mask), pred_mask))
    precision = true_positive / (true_positive + false_positive + 1e-6)
    return precision

def calculate_f1_score(recall, precision):
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1_score

Label = namedtuple('Label', [ 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color',])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'ignore'               , -1 ,       19 , 'void'            , 0       , False        , True         , (  0,  0,142) ),
]

category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

def generateMatrix():
    evalLabels = []
    for label in labels:
        evalLabels.append(label.trainId)
    maxId = max(evalLabels)
    return evalLabels, np.zeros(shape=(maxId+1, maxId+1),dtype=np.ulonglong)

def sampleEvaluate(pred_mask, gt_mask, evalLabels, confMatrix):
    predictionNp  = np.array(pred_mask)
    groundTruthNp = np.array(gt_mask)

    imgWidth  = predictionNp.shape[0]
    imgHeight = predictionNp.shape[1]
    nbPixels  = imgWidth*imgHeight

    if (CSUPPORT):
        confMatrix = addToConfusionMatrix.cEvaluatePair(predictionNp, groundTruthNp, confMatrix, evalLabels)
    else:
        encoding_value = max(groundTruthNp.max(), predictionNp.max()).astype(np.int32) + 1
        encoded = (groundTruthNp.astype(np.int32) * encoding_value) + predictionNp

        values, cnt = np.unique(encoded, return_counts=True)

        for value, c in zip(values, cnt):
            pred_id = value % encoding_value
            gt_id = int((value - pred_id)/encoding_value)
            if not gt_id in evalLabels:
                printError("Unknown label with id {:}".format(gt_id))
            confMatrix[gt_id][pred_id] += c

    return nbPixels


def SS_calc_metric(gt_mask, pred_mask):
    pred_mask[pred_mask >= 0.3] = 1
    pred_mask[pred_mask < 0.3] = 0

    gt_mask = gt_mask.astype(int).flatten()
    pred_mask = pred_mask.astype(int).flatten()

    #intersection = np.sum(gt_mask * pred_mask)
    #iou = intersection / (np.sum(gt_mask + pred_mask <= 2) + 1e-6)
    #prec = intersection / (np.sum(pred_mask == 1) + 1e-6)
    #reca = intersection / (np.sum(gt_mask == 1) + 1e-6)
    #f_one = 2 * prec * reca / (prec + reca + 1e-6)

    iou = calculate_iou(gt_mask, pred_mask)
    prec = calculate_precision(gt_mask, pred_mask)
    reca = calculate_recall(gt_mask, pred_mask)
    f_one = calculate_f1_score(prec, reca)

    return iou, prec, reca, f_one

def getMetricsForLabel(label, confMatrix, evalLabels):
    tp = np.longlong(confMatrix[label,label])

    fn = np.longlong(confMatrix[label,:].sum()) - tp

    notIgnored = [l for l in evalLabels if not labels[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    denom = (tp + fp + fn + 1e-6)

    iou = float(tp) / denom
    pre = float(tp) / (tp + fp + 1e-6)
    rec = float(tp) / (tp + fn + 1e-6)
    f1 = 2 * pre * rec / (pre + rec + 1e-6)

    return iou, pre, rec, f1

def getMetricsForCategory(category, confMatrix, evalLabels):
    labels_in = category2labels[category]
    labelIds = [label.trainId for label in labels_in if not label.ignoreInEval and label.trainId in evalLabels]

    tp = np.longlong(confMatrix[labelIds,:][:,labelIds].sum())

    fn = np.longlong(confMatrix[labelIds,:].sum()) - tp

    notIgnoredAndNotInCategory = [l for l in evalLabels if not labels[l].ignoreInEval and labels[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    denom = (tp + fp + fn + 1e-6)
    iou = float(tp) / denom
    pre = float(tp) / (tp + fp + 1e-6)
    rec = float(tp) / (tp + fn + 1e-6)
    f1 = 2 * pre * rec / (pre + rec + 1e-6)

    return iou, pre, rec, f1

def getMetricsAvg(scoreList):
    validScores = 0
    scoreSum    = 0.0
    iouSum = 0.0
    preSum = 0.0
    recSum = 0.0
    f1Sum = 0.0
    for label, metric in scoreList.items():
        if label == 'ignore' or label == 'void':
            continue
        validScores += 1
        iouSum += metric['IoU']
        preSum += metric['Precision']
        recSum += metric['Recall']
        f1Sum += metric['F1']
    avg_iou = iouSum / (validScores + 1e-6)
    avg_pre = preSum / (validScores + 1e-6)
    avg_rec = recSum / (validScores + 1e-6)
    avg_f1 = f1Sum / (validScores + 1e-6)
    return avg_iou, avg_pre, avg_rec, avg_f1

def SS_Evaluate(model, dataloader, dev):
    model.eval()
    model.aux_mode = 'test'
    cla_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafficlight', 'trafficsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'ignore']

    eval_loss = 0.0
    testdatalen = len(dataloader)
    with torch.no_grad():
        evalLabels, confMatrix = generateMatrix()
        nbPixels = 0
        for imgs, masks, names in tqdm(dataloader):
            _masks = F.softmax(model(imgs.to(dev)), dim=1)
            eval_loss += F.cross_entropy(_masks, masks.to(_masks.device), ignore_index=19)
            max_prob, pred_masks = torch.max(_masks, dim=1)

            masks = masks.squeeze().detach().cpu().numpy()
            pred_masks = pred_masks.squeeze().detach().cpu().numpy()
            for i in range(masks.shape[0]):
                nbPixels += sampleEvaluate(pred_masks[i], masks[i], evalLabels, confMatrix)

        if confMatrix.sum() != nbPixels:
            printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

        clsScoreList = {}
        for label in evalLabels:
            classScoreList = {}
            labelName = cla_names[label]
            iou, pre, recall, f1 = getMetricsForLabel(label, confMatrix, evalLabels)

            classScoreList['IoU'] = iou
            classScoreList['Precision'] = pre
            classScoreList['Recall'] = recall
            classScoreList['F1'] = f1
            clsScoreList[labelName] = classScoreList

        miou, mpre, mrec, mf1 = getMetricsAvg(clsScoreList)
        clsScoreList['mIoU'] = miou
        clsScoreList['mPrecision'] = mpre
        clsScoreList['mRecall'] = mrec
        clsScoreList['mF1'] = mf1

        catScoreList = {}
        for category in category2labels.keys():
            categoryScoreList = {}
            iou, pre, recall, f1 = getMetricsForCategory(category, confMatrix, evalLabels)

            categoryScoreList['IoU'] = iou
            categoryScoreList['Precision'] = pre
            categoryScoreList['Recall'] = recall
            categoryScoreList['F1'] = f1
            catScoreList[category] = categoryScoreList

        miou, mpre, mrec, mf1 = getMetricsAvg(catScoreList)
        catScoreList['mIoU'] = miou
        catScoreList['mPrecision'] = mpre
        catScoreList['mRecall'] = mrec
        catScoreList['mF1'] = mf1

    model.train()
    return clsScoreList, catScoreList, eval_loss / testdatalen

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
