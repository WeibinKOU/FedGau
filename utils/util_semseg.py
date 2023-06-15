from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from utils.utils_cla import progress_bar
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

def SS_calc_IoU(gt_mask, pred_mask):

    iou = jaccard(gt_mask.flatten(), pred_mask.flatten())
    prec = precision(gt_mask.flatten(), pred_mask.flatten())
    reca = recall(gt_mask.flatten(), pred_mask.flatten())
    f_one = f1(gt_mask.flatten(), pred_mask.flatten())

    return iou.item(), prec.item(), reca.item(), f_one.item()

def SS_Evaluate(model, dataloader, dev):
    model.eval()
    model.aux_mode = 'test'

    with torch.no_grad():
        iou = []
        precision = []
        recall = []
        f1 = []
        for imgs, imgs_inv, masks, names in tqdm(dataloader):
            pred_masks = F.softmax(model(imgs.to(dev), imgs_inv.to(dev)), dim=1)[:, 0, :, :]
            t = pred_masks
            t[t >= 0.3] = 1.0
            t[t < 0.3] = 0.0
            pred_masks = t
            masks = masks.squeeze().numpy().astype(int)
            pred_masks = pred_masks.squeeze().detach().cpu().numpy().astype(int)
            for i in range(masks.shape[0]):
                iou_t, pre_t, rec_t, f1_t = SS_calc_IoU(masks[i], pred_masks[i])
                iou.append(iou_t)
                precision.append(pre_t)
                recall.append(rec_t)
                f1.append(f1_t)
        avg_iou = sum(iou) / len(iou)
        avg_pre = sum(precision) / len(precision)
        avg_recall = sum(recall) / len(recall)
        avg_f1 = sum(f1) / len(f1)
    model.train()
    return avg_iou, avg_pre, avg_recall, avg_f1 



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
