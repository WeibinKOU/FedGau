import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

import sys
sys.path.append("../")

from model.bisenetv2 import BiSeNetV2
from model.cracknet import CrackNet
from model.segnet import SegNet
from model.deeplabv3 import DeepLabv3

from data import Dataset 
from config import *
from utils.average_meter import AverageMeter

class Client():
    def __init__(self, server_id, client_id, global_idx, config, test_data, aug_seq, clients_dict, training_config, tensorboard, scheduler, model_sel, dev):
        self.sid = server_id
        self.cid = client_id
        self.global_idx = global_idx
        self.tb = tensorboard
        self.epochs = training_config['epochs']
        self.batch_size = training_config['batch_size']
        self.clients_dict = clients_dict
        self.model = model_sel['model'](n_classes=2).to(dev)
        self.updated_model = None
        self.epoch_cnt = 0
        self.eval_data = test_data
        self.fed_cnt = 0
        self.criterion = nn.BCELoss().to(dev)
        self.scheduler = scheduler
        self.dev = dev

        self.dataset = Dataset(config['dataset'], transform=aug_seq)

        self.dataloader = DataLoader(self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1)
        self.data_len = len(self.dataloader) * self.batch_size

        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=training_config['lr'],
                                      betas=training_config['betas'],
                                      weight_decay=training_config['weight_decay'])


        self.batch_cnt = len(self.dataloader)


    def PreparePretrainData(self):
        data = {}
        data['loader'] = self.dataloader
        entries_cnt = self.scheduler.pretrain_batch_cnt[self.global_idx] * self.batch_size

        self.scheduler.wireless_stat(entries_cnt * self.scheduler.entry_size)
        ret = self.scheduler.transfer_entries(entries_cnt)
        if not ret:
            print("Initialized data size is not enough to transfer pretaining data, so exit!")
            sys.exit()
        data['entries_cnt'] = entries_cnt

        return data

    def PrepareTrainData(self):
        data = {}
        data['loader'] = self.dataloader
        return data

    def calc_IOU(self, gt_mask, pred_mask):
        from sklearn.metrics import jaccard_score as jaccard
        from sklearn.metrics import precision_score as precision
        from sklearn.metrics import recall_score as recall
        from sklearn.metrics import f1_score as f1

        iou = jaccard(gt_mask.flatten(), pred_mask.flatten())
        prec = precision(gt_mask.flatten(), pred_mask.flatten())
        recall = recall(gt_mask.flatten(), pred_mask.flatten())
        f_one = f1(gt_mask.flatten(), pred_mask.flatten())

        return iou.item(), prec.item(), recall.item(), f_one.item()

    def Evaluate(self):
        self.model.eval()
        self.model.aux_mode = 'test'

        dataloader = self.eval_data

        with torch.no_grad():
            iou = []
            precision = []
            recall = []
            f1 = []
            for imgs, imgs_inv, masks, names in tqdm(dataloader):
                pred_masks = F.softmax(self.model(imgs.to(self.dev), imgs_inv.to(self.dev)), dim=1)[:, 0, :, :]
                t = pred_masks
                t[t >= 0.3] = 1.0
                t[t < 0.3] = 0.0
                pred_masks = t
                masks = masks.squeeze().numpy().astype(int)
                pred_masks = pred_masks.squeeze().detach().cpu().numpy().astype(int)
                for i in range(masks.shape[0]):
                    iou_t, pre_t, rec_t, f1_t = self.calc_IOU(masks[i], pred_masks[i])
                    iou.append(iou_t)
                    precision.append(pre_t)
                    recall.append(rec_t)
                    f1.append(f1_t)
            avg_iou = sum(iou) / len(iou)
            avg_pre = sum(precision) / len(precision)
            avg_recall = sum(recall) / len(recall)
            avg_f1 = sum(f1) / len(f1)
        return avg_iou, avg_pre, avg_recall, avg_f1 

    def train(self):
        for epoch in range(self.scheduler.edge_fed_interval):
            if self.updated_model is not None:
                self.model.load_state_dict(self.updated_model)
                self.updated_model = None

            self.model.train()
            self.model.aux_mode = 'train'

            loss = AverageMeter()
            for imgs, imgs_inv, masks in tqdm(self.dataloader):
                *logits, = self.model(imgs.to(self.dev), imgs_inv.to(self.dev))
                pred_masks = [F.softmax(logit, dim=1) for logit in logits]
                dist = [self.criterion(pred_mask, masks.to(self.dev)) for pred_mask in pred_masks]
                dist = sum(dist)
                self.optim.zero_grad()
                dist.backward()
                loss.update(dist.item())
                self.optim.step()

            cid = "%s.%s" % (self.sid, self.cid)
            log_info = "[Epoch: %d/%d] [%s.Train.Loss: %f]" % (self.epoch_cnt, self.epochs, cid, loss.avg)
            print(log_info)

            self.tb.add_scalar(cid + '.Train.Loss', loss.avg, self.epoch_cnt)

            '''
            if self.epoch_cnt % 5 == 4:
                avg_iou, avg_pre, avg_recall, avg_f1 = self.Evaluate()
                self.tb.add_scalar(cid + '.Eval.AvgIOU', avg_iou, self.epoch_cnt)
                self.tb.add_scalar(cid + '.Eval.AvgPrecision', avg_pre, self.epoch_cnt)
                self.tb.add_scalar(cid + '.Eval.AvgRecall', avg_recall, self.epoch_cnt)
                self.tb.add_scalar(cid + '.Eval.AvgF1', avg_f1, self.epoch_cnt)
                log_info = "[Epoch: %d/%d] [%s.Eval.AvgIOU: %f, %s.Eval.AvgPrecision: %f, %s.Eval.AvgRecall: %f, %s.Eval.AvgF1: %f]" % (self.epoch_cnt, self.epochs, cid, avg_iou, cid, avg_pre, cid, avg_recall, cid, avg_f1)
                print(log_info)
            '''

            self.epoch_cnt += 1

            if epoch == self.scheduler.edge_fed_interval - 1:
                self.clients_dict[self.cid] = self.model.state_dict()
                self.scheduler.wireless_stat(self.scheduler.model_size)
                ret = self.scheduler.transfer_model()
                if not ret:
                    print("Initialized data size has been used up, so exit!")

    def pretrain(self):
        for epoch in range(self.scheduler.pretrain_epochs):
            self.model.train()
            self.model.aux_mode = 'train'

            loss = AverageMeter()
            for imgs, imgs_inv, masks in tqdm(self.dataloader):
                *logits, = self.model(imgs.to(self.dev), imgs_inv.to(self.dev))
                pred_masks = [F.softmax(logit, dim=1) for logit in logits]
                dist = [self.criterion(pred_mask, masks.to(self.dev)) for pred_mask in pred_masks]
                dist = sum(dist)
                self.optim.zero_grad()
                dist.backward()
                loss.update(dist.item())
                self.optim.step()

            cid = "%s.%s" % (self.sid, self.cid)
            log_info = "[Pretrain] [Epoch: %d/%d] [%s.Pretrain.Loss: %f]" % (epoch, self.scheduler.pretrain_epochs, cid, loss.avg)
            print(log_info)

            self.tb.add_scalar(cid + '.Pretrain.Loss', loss.avg, epoch)
