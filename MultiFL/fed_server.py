import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F
import os
from tqdm import tqdm
import imgaug.augmenters as iaa
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1

from .fed_client import Client
from .fed_scheduler import Scheduler

import sys
sys.path.append("../")

from model.bisenetv2 import BiSeNetV2
from model.cracknet import CrackNet
from model.segnet import SegNet
from model.deeplabv3 import DeepLabv3

from data import Dataset 
from config import *

class EdgeServer():
    client_global_idx = 0
    def __init__(self, server_id, config, test_data, aug_seq, edges_dict, training_config, tensorboard, scheduler, model_sel, dev):
        self.id = server_id
        self.clients_num = len(config)
        self.clients_dict = {}
        self.avgModel = None
        self.edges_dict = edges_dict
        self.epochs = training_config['epochs']
        self.batch_size = training_config['batch_size']
        self.model = model_sel['model'](n_classes=2).to(dev).eval()
        self.tb = tensorboard
        self.eval_data = test_data
        self.fed_cnt = 0
        self.criterion = nn.BCELoss().to(dev)
        self.scheduler = scheduler
        self.dev = dev

        self.clients = []
        for i in range(self.clients_num):
            cid = 'Agent'+str(i)
            self.clients.append(Client(self.id, cid, self.__class__.client_global_idx, config[cid], self.eval_data, aug_seq, self.clients_dict, training_config, tensorboard, scheduler, model_sel, dev))
            print('%s.%s has been initialized!' % (self.id, cid))
            self.__class__.client_global_idx += 1

    def run(self):
        for i in range(self.clients_num):
            self.clients[i].train()

    def FedAvg(self):
        w_avg = copy.deepcopy(self.clients_dict['Agent0'])
        for k in w_avg.keys():
            for i in range(1, len(self.clients_dict)):
                w_avg[k] += self.clients_dict['Agent' + str(i)][k]
            w_avg[k] = torch.div(w_avg[k], len(self.clients_dict))

        self.avgModel = w_avg

        avg_iou, avg_pre, avg_recall, avg_f1 = self.Evaluate()
        self.tb.add_scalar('%s.Eval.AvgIOU' % self.id, avg_iou, self.fed_cnt)
        self.tb.add_scalar('%s.Eval.AvgPrecision' % self.id, avg_pre, self.fed_cnt)
        self.tb.add_scalar('%s.Eval.AvgRecall' % self.id, avg_recall, self.fed_cnt)
        self.tb.add_scalar('%s.Eval.AvgF1' % self.id, avg_f1, self.fed_cnt)
        log_info = "[Edge FL: %d] [%s.FL.Eval.AvgIOU: %f, %s.FL.Eval.AvgPrecision: %f, %s.FL.Eval.AvgRecall: %f, %s.FL.Eval.AvgF1: %f]" % (self.fed_cnt, self.id, avg_iou, self.id, avg_pre, self.id, avg_recall, self.id, avg_f1)
        print(log_info)

        self.fed_cnt += 1

        self.edges_dict[self.id] = self.avgModel
        self.scheduler.wireline_stat(self.scheduler.model_size)
        ret = self.scheduler.transfer_model()
        if not ret:
            print("Initialized data size has been used up, so exit!")
            sys.exit()

    def SinkModelToClients(self):
        for client in self.clients:
            client.updated_model = self.avgModel
            self.scheduler.wireless_stat(self.scheduler.model_size)
            ret = self.scheduler.transfer_model()
            if not ret:
                print("Initialized data size has been used up, so exit!")
                sys.exit()

    def PreparePretrainData(self):
        data = []
        entries_cnt = 0
        for client in self.clients:
            cdata = client.PreparePretrainData()
            entries_cnt += cdata['entries_cnt']
            data.append(cdata)

        self.scheduler.wireline_stat(entries_cnt * self.scheduler.entry_size)
        ret = self.scheduler.transfer_entries(entries_cnt)
        if not ret:
            print("Initialized data size is not enough to transfer pretaining data, so exit!")
            sys.exit()

        return data

    def PrepareTrainData(self):
        data = []
        for client in self.clients:
            data.append(client.PrepareTrainData())

        return data

    def calc_IOU(self, gt_mask, pred_mask):

        iou = jaccard(gt_mask.flatten(), pred_mask.flatten())
        prec = precision(gt_mask.flatten(), pred_mask.flatten())
        reca = recall(gt_mask.flatten(), pred_mask.flatten())
        f_one = f1(gt_mask.flatten(), pred_mask.flatten())

        return iou.item(), prec.item(), reca.item(), f_one.item()

    def Evaluate(self):
        self.model.load_state_dict(self.avgModel)
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

class CloudServer():
    def __init__(self, aug_seq, training_config, tensorboard, scheduler, logdir, model_sel, dev):
        from .fed_config import config
        self.edges_num = len(config) - 1
        self.edges_dict = {}
        self.pretrain_data = None
        self.model = model_sel['model'](n_classes=2).to(dev)
        if os.path.exists('./checkpoints/init_model.pth'):
            self.model.load_state_dict(torch.load('./checkpoints/init_model.pth', map_location=torch.device(dev)))
        self.avgModel = self.model.state_dict()
        self.pretrain_config = training_config
        self.tb = tensorboard
        self.epochs = training_config['epochs']
        self.logdir = logdir
        self.dev = dev

        self.criterion = nn.BCELoss().to(dev)
        self.fed_cnt = 0

        def build_eval_data(conf):
            seq = iaa.Sequential([
                iaa.Resize({'height': HEIGHT, 'width': WIDTH}),
            ])
            dataset = Dataset(conf['test']['dataset'], seq, type_='test')

            dataloader = DataLoader(dataset,
                    batch_size=self.pretrain_config['batch_size'],
                    #batch_size=1,
                    shuffle=True,
                    num_workers=1,
                    drop_last=True)

            return dataloader

        self.eval_data = build_eval_data(config)

        self.scheduler = scheduler

        self.edges = []
        for i in range(self.edges_num):
            eid = 'Edge'+str(i)
            self.edges.append(EdgeServer(eid, config[eid], self.eval_data, aug_seq, self.edges_dict,
                                         training_config, tensorboard, self.scheduler, model_sel, dev))
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.pretrain_config['lr'], betas=self.pretrain_config['betas'], weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=20, gamma=0.1)

    def run(self):
        save_path = MODEL_DIR + self.logdir
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        edge_fed_cnt = 0
        for j in range(self.scheduler.epochs_after_pretrain):
            for edge in self.edges:
                edge.run()

            edge_fed_cnt += 1
            if edge_fed_cnt == self.scheduler.cloud_fed_interval:
                edge_fed_cnt = 0
                self.FedAvg()
                if j % 10 == 9:
                    save_name = save_path + "/Cloud_FL_%d_model.pth" % j
                    torch.save(self.avgModel, save_name)

                self.SinkModelToEdges()
                for edge in self.edges:
                    edge.SinkModelToClients()
                    if j % 10 == 9:
                        save_name = save_path + "/%s_%d_FL_model.pth" % (edge.id, j)
                        torch.save(edge.avgModel, save_name)
            else:
                for edge in self.edges:
                    edge.FedAvg()
                    edge.SinkModelToClients()
                    if j % 10 == 9:
                        save_name = save_path + "/%s_%d_FL_model.pth" % (edge.id, j)
                        torch.save(edge.avgModel, save_name)

            self.tb.add_scalar('Scheduler.Wireless.Size', self.scheduler.wireless_size, j)
            self.tb.add_scalar('Scheduler.Wireline.Size', self.scheduler.wireline_size, j)

    def FedAvg(self):
        for edge in self.edges:
            edge.FedAvg()

        w_avg = copy.deepcopy(self.edges_dict['Edge0'])
        for k in w_avg.keys():
            for i in range(1, len(self.edges_dict)):
                w_avg[k] += self.edges_dict['Edge' + str(i)][k]
            w_avg[k] = torch.div(w_avg[k], len(self.edges_dict))

        self.avgModel = w_avg

        self.model.load_state_dict(self.avgModel)

        avg_iou, avg_pre, avg_recall, avg_f1 = self.Evaluate()
        self.tb.add_scalar('Cloud.FL.Eval.AvgIOU', avg_iou, self.fed_cnt)
        self.tb.add_scalar('Cloud.FL.Eval.AvgPrecision', avg_pre, self.fed_cnt)
        self.tb.add_scalar('Cloud.FL.Eval.AvgRecall', avg_recall, self.fed_cnt)
        self.tb.add_scalar('Cloud.FL.Eval.AvgF1', avg_f1, self.fed_cnt)
        log_info = "[Cloud FL: %d] [Cloud.FL.Eval.AvgIOU: %f, Cloud.FL.Eval.AvgPrecision: %f, Cloud.FL.Eval.AvgRecall: %f, Cloud.FL.Eval.AvgF1: %f]" % (self.fed_cnt, avg_iou, avg_pre, avg_recall, avg_f1)
        print(log_info)

        self.fed_cnt += 1

    def SinkModelToEdges(self):
        for edge in self.edges:
            edge.avgModel = self.avgModel
            self.scheduler.wireline_stat(self.scheduler.model_size)
            ret = self.scheduler.transfer_model()
            if not ret:
                print("Initialized data size has been used up, so exit!")
                sys.exit()

    def calc_IOU(self, gt_mask, pred_mask):
        iou = jaccard(gt_mask.flatten(), pred_mask.flatten())
        prec = precision(gt_mask.flatten(), pred_mask.flatten())
        reca = recall(gt_mask.flatten(), pred_mask.flatten())
        f_one = f1(gt_mask.flatten(), pred_mask.flatten())

        return iou.item(), prec.item(), reca.item(), f_one.item()

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
