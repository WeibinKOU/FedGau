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

cityscapes_labels = [
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

mapillary_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'Bird'                 ,  0 ,        0 , 'animal'          , 0       , True         , False         , (165, 42, 42) ),
    Label(  'Ground Animal'        ,  1 ,        1 , 'animal'          , 0       , True         , False         , (0, 192, 0) ),
    Label(  'Curb'                 ,  2 ,        2 , 'construction'    , 1       , False        , False         , (196, 196, 196) ),
    Label(  'Fence'                ,  3 ,        3 , 'construction'    , 1       , False        , False         , (190, 153, 153) ),
    Label(  'Guard Rail'           ,  4 ,        4 , 'construction'    , 1       , False        , False         , (180, 165, 180) ),
    Label(  'Barrier'              ,  5 ,        5 , 'construction'    , 1       , False        , False         , (90, 120, 150)) ,
    Label(  'Wall'                 ,  6 ,        6 , 'construction'    , 1       , False        , False         , (102, 102, 156)),
    Label(  'Bike Lane'            ,  7 ,        7 , 'construction'    , 1       , False        , False         , (128, 64, 255)),
    Label(  'Crosswalk - Plain'    ,  8 ,        8 , 'construction'    , 1       , True         , False         , (140, 140, 200)),
    Label(  'Curb Cut'             ,  9 ,        9 , 'construction'    , 1       , False        , False         , (170, 170, 170)),
    Label(  'Parking'              , 10 ,       10 , 'construction'    , 1       , False        , False         , (250, 170, 160)),
    Label(  'Pedestrian Area'      , 11 ,       11 , 'construction'    , 1       , False        , False         , (96, 96, 96)),
    Label(  'Rail Track'           , 12 ,       12 , 'construction'    , 1       , False        , False         , (230, 150, 140)),
    Label(  'Road'                 , 13 ,       13 , 'construction'    , 1       , False        , False         , (128, 64, 128)),
    Label(  'Service Lane'         , 14 ,       14 , 'construction'    , 1       , False        , False         , (110, 110, 110)),
    Label(  'Sidewalk'             , 15 ,       15 , 'construction'    , 1       , False        , False         , (244, 35, 232)),
    Label(  'Bridge'               , 16 ,       16 , 'construction'    , 1       , False        , False         , (150, 100, 100)),
    Label(  'Building'             , 17 ,       17 , 'construction'    , 1       , False        , False         , (70, 70, 70)),
    Label(  'Tunnel'               , 18 ,       18 , 'construction'    , 1       , False        , False         , (150, 120, 90)),
    Label(  'Person'               , 19 ,       19 , 'human'           , 2       , True         , False         , (220, 20, 60)),
    Label(  'Bicyclist'            , 20 ,       20 , 'human'           , 2       , True         , False         , (255, 0, 0)),
    Label(  'Motorcyclist'         , 21 ,       21 , 'human'           , 2       , True         , False         , (255, 0, 100)),
    Label(  'Other Rider'          , 22 ,       22 , 'human'           , 2       , True         , False         , (255, 0, 200)),
    Label(  'LaneMarking-Crosswalk', 23 ,       23 , 'marking'         , 3       , True         , False         , (200, 128, 128)),
    Label(  'LaneMarking-General'  , 24 ,       24 , 'marking'         , 3       , False        , False         , (255, 255, 255)),
    Label(  'Mountain'             , 25 ,       25 , 'nature'          , 4       , False        , False         , (64, 170, 64)),
    Label(  'Sand'                 , 26 ,       26 , 'nature'          , 4       , False        , False         , (230, 160, 50)),
    Label(  'Sky'                  , 27 ,       27 , 'nature'          , 4       , False        , False         , (70, 130, 180)),
    Label(  'Snow'                 , 28 ,       28 , 'nature'          , 4       , False        , False         , (190, 255, 255)),
    Label(  'Terrain'              , 29 ,       29 , 'nature'          , 4       , False        , False         , (152, 251, 152)),
    Label(  'Vegetation'           , 30 ,       30 , 'nature'          , 4       , False        , False         , (107, 142, 35)),
    Label(  'Water'                , 31 ,       31 , 'nature'          , 4       , False        , False         , (0, 170, 30)),
    Label(  'Banner'               , 32 ,       32 , 'object'          , 5       , True         , False         , (255, 255, 128)),
    Label(  'Bench'                , 33 ,       33 , 'object'          , 5       , True         , False         , (250, 0, 30)),
    Label(  'Bike Rack'            , 34 ,       34 , 'object'          , 5       , True         , False         , (100, 140, 180)),
    Label(  'Billboard'            , 35 ,       35 , 'object'          , 5       , True         , False         , (220, 220, 220)),
    Label(  'Catch Basin'          , 36 ,       36 , 'object'          , 5       , True         , False         , (220, 128, 128)),
    Label(  'CCTV Camera'          , 37 ,       37 , 'object'          , 5       , True         , False         , (222, 40, 40)),
    Label(  'Fire Hydrant'         , 38 ,       38 , 'object'          , 5       , True         , False         , (100, 170, 30)),
    Label(  'Junction Box'         , 39 ,       39 , 'object'          , 5       , True         , False         , (40, 40, 40)),
    Label(  'Mailbox'              , 40 ,       40 , 'object'          , 5       , True         , False         , (33, 33, 33)),
    Label(  'Manhole'              , 41 ,       41 , 'object'          , 5       , True         , False         , (100, 128, 160)),
    Label(  'Phone Booth'          , 42 ,       42 , 'object'          , 5       , True         , False         , (142, 0, 0)),
    Label(  'Pothole'              , 43 ,       43 , 'object'          , 5       , False        , False         , (70, 100, 150)),
    Label(  'Street Light'         , 44 ,       44 , 'object'          , 5       , True         , False         , (210, 170, 100)),
    Label(  'Pole'                 , 45 ,       45 , 'object'          , 5       , True         , False         , (153, 153, 153)),
    Label(  'Traffic Sign Frame'   , 46 ,       46 , 'object'          , 5       , True         , False         , (128, 128, 128)),
    Label(  'Utility Pole'         , 47 ,       47 , 'object'          , 5       , True         , False         , (0, 0, 80)),
    Label(  'Traffic Light'        , 48 ,       48 , 'object'          , 5       , True         , False         , (250, 170, 30)),
    Label(  'Traffic Sign (Back)'  , 49 ,       49 , 'object'          , 5       , True         , False         , (192, 192, 192)),
    Label(  'Traffic Sign (Front)' , 50 ,       50 , 'object'          , 5       , True         , False         , (220, 220, 0)),
    Label(  'Trash Can'            , 51 ,       51 , 'object'          , 5       , True         , False         , (140, 140, 20)),
    Label(  'Bicycle'              , 52 ,       52 , 'object'          , 5       , True         , False         , (119, 11, 32)),
    Label(  'Boat'                 , 53 ,       53 , 'object'          , 5       , True         , False         , (150, 0, 255)),
    Label(  'Bus'                  , 54 ,       54 , 'object'          , 5       , True         , False         , (0, 60, 100)),
    Label(  'Car'                  , 55 ,       55 , 'object'          , 5       , True         , False         , (0, 0, 142)),
    Label(  'Caravan'              , 56 ,       56 , 'object'          , 5       , True         , False         , (0, 0, 90)),
    Label(  'Motorcycle'           , 57 ,       57 , 'object'          , 5       , True         , False         , (0, 0, 230)),
    Label(  'On Rails'             , 58 ,       58 , 'object'          , 5       , False        , False         , (0, 80, 100)),
    Label(  'Other Vehicle'        , 59 ,       59 , 'object'          , 5       , True         , False         , (128, 64, 64)),
    Label(  'Trailer'              , 60 ,       60 , 'object'          , 5       , True         , False         , (0, 0, 110)),
    Label(  'Truck'                , 61 ,       61 , 'object'          , 5       , True         , False         , (0, 0, 70)),
    Label(  'Wheeled Slow'         , 62 ,       62 , 'object'          , 5       , True         , False         , (0, 0, 192)),
    Label(  'Car Mount'            , 63 ,       63 , 'object'          , 5       , False        , False         , (32, 32, 32)),
    Label(  'Ego Vehicle'          , 64 ,       64 , 'object'          , 5       , False        , False         , (120, 10, 10)),
    Label(  'ignore'               , 65 ,       65 , 'void'            , 6       , False        , True          , (0, 0, 0))
    ]


CamVid_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'sky'                  ,  0 ,        0 , 'sky'             ,    0     , False        , False        , (128, 64,128) ),
    Label(  'building'             ,  1 ,        1 , 'building'        ,    1     , False        , False        , (244, 35,232) ),
    Label(  'pole'                 ,  2 ,        2 , 'pole'            ,    2     , False        , False        , ( 70, 70, 70) ),
    Label(  'road'                 ,  3 ,        3 , 'road'            ,    3     , False        , False        , (102,102,156) ),
    Label(  'sidewalk'             ,  4 ,        4 , 'sidewalk'        ,    4     , False        , False        , (190,153,153) ),
    Label(  'tree'                 ,  5 ,        5 , 'tree'            ,    5     , False        , False        , (153,153,153) ),
    Label(  'signsymbol'           ,  6 ,        6 , 'signsymbol'      ,    6     , False        , False        , (250,170, 30) ),
    Label(  'fence'                ,  7 ,        7 , 'fence'           ,    7     , False        , False        , (220,220,  0) ),
    Label(  'car'                  ,  8 ,        8 , 'car'             ,    8     , False        , False        , (107,142, 35) ),
    Label(  'pedestrian'           ,  9 ,        9 , 'pedestrian'      ,    9     , False        , False        , (152,251,152) ),
    Label(  'bicyclist'            , 10 ,       10 , 'bicyclist'       ,   10     , False        , False        , ( 70,130,180) ),
    Label(  'ignore'               , 11 ,       11 , 'void'            ,   11     , False        , True         , (  0,  0,142) ),
]

CARLA_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'ignore'           ,      0 ,      0   ,   'ignore'         ,    0     , False        , True         , (  0,  0,  0) ),
    Label(  'building'         ,      1 ,      1   ,   'building'       ,    1     , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'            ,      2 ,      2   ,   'fence'          ,    2     , False        , False        , (100, 40, 40) ),
    Label(  'other'            ,      3 ,      3   ,   'other'          ,    3     , False        , False        , ( 55, 90, 80) ),
    Label(  'pedestrian'       ,      4 ,      4   ,   'pedestrian'     ,    4     , False        , False        , (220, 20, 60) ),
    Label(  'pole'             ,      5 ,      5   ,   'pole'           ,    5     , False        , False        , (153,153,153) ),
    Label(  'roadline'         ,      6 ,      6   ,   'roadline'       ,    6     , False        , False        , (157,234, 50) ),
    Label(  'road'             ,      7 ,      7   ,   'road'           ,    7     , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'         ,      8 ,      8   ,   'sidewalk'       ,    8     , False        , False        , (244, 35,232) ),
    Label(  'vegetation'       ,      9 ,      9   ,   'vegetation'     ,    9     , False        , False        , (107,142, 35) ),
    Label(  'vehicles'         ,     10 ,     10   ,   'vehicles'       ,   10     , False        , False        , (  0,  0,142) ),
    Label(  'wall'             ,     11 ,     11   ,   'wall'           ,   11     , False        , False        , (102,102,156) ),
    Label(  'trafficsign'      ,     12 ,     12   ,   'trafficsign'    ,   12     , False        , False        , (220,220,  0) ),
    Label(  'sky'              ,     13 ,     13   ,   'sky'            ,   13     , False        , False        , ( 70,130,180) ),
    Label(  'ground'           ,     14 ,     14   ,   'ground'         ,   14     , False        , False        , ( 81,  0, 81) ),
    Label(  'bridge'           ,     15 ,     15   ,   'bridge'         ,   15     , False        , False        , (150,100,100) ),
    Label(  'railtrack'        ,     16 ,     16   ,   'railtrack'      ,   16     , False        , False        , (230,150,140) ),
    Label(  'guardrail'        ,     17 ,     17   ,   'guardrail'      ,   17     , False        , False        , (180,165,180) ),
    Label(  'trafficlight'     ,     18 ,     18   ,   'trafficlight'   ,   18     , False        , False        , (250,170, 30) ),
    Label(  'static'           ,     19 ,     19   ,   'static'         ,   19     , False        , False        , (110,190,160) ),
    Label(  'dynamic'          ,     20 ,     20   ,   'dynamic'        ,   20     , False        , False        , (170,120, 50) ),
    Label(  'water'            ,     21 ,     21   ,   'water'          ,   21     , False        , False        , ( 45, 60,150) ),
    Label(  'terrain'          ,     22 ,     22   ,   'terrain'        ,   22     , False        , False        , (145,170,100) ),
]


def generateMatrix(labels):
    category2labels = {}
    evalLabels = []
    for label in labels:
        evalLabels.append(label.trainId)
        category = label.category
        if category in category2labels:
            category2labels[category].append(label)
        else:
            category2labels[category] = [label]
    maxId = max(evalLabels)
    return evalLabels, np.zeros(shape=(maxId+1, maxId+1), dtype=np.ulonglong), category2labels

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

def getMetricsForLabel(label, confMatrix, evalLabels, labels):
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

def getMetricsForCategory(category, confMatrix, evalLabels, labels, category2labels):
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

def SS_Evaluate(model, dataloader, dev, dataset_name):
    model.eval()
    model.aux_mode = 'test'
    if dataset_name == 'cityscapes':
        cla_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafficlight', 'trafficsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'ignore']
        labels = cityscapes_labels
    elif dataset_name == 'Mapillary':
        cla_names = ['Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall', 'Bike Lane', 'Crosswalk-Plain', 'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'LaneMarking-Crosswalk', 'LaneMarking-General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole', 'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'ignore']
        labels = mapillary_labels
    elif dataset_name == 'CamVid':
        cla_names = ['sky', 'building', 'pole', 'road', 'sidewalk', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'ignore']
        labels = CamVid_labels
    elif dataset_name == 'CARLA':
        cla_names = ['ignore', 'building', 'fence', 'other', 'pedestrian', 'pole', 'roadline', 'road', 'sidewalk', 'vegetation', 'vehicles', 'wall', 'trafficsign', 'sky', 'ground', 'bridge', 'railtrack', 'guardrail', 'trafficlight', 'static', 'dynamic', 'water', 'terrain']
        labels = CARLA_labels

    eval_loss = 0.0
    testdatalen = len(dataloader)
    with torch.no_grad():
        evalLabels, confMatrix, category2labels = generateMatrix(labels)
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
            iou, pre, recall, f1 = getMetricsForLabel(label, confMatrix, evalLabels, labels)

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
            iou, pre, recall, f1 = getMetricsForCategory(category, confMatrix, evalLabels, labels, category2labels)

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
