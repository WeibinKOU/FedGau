import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import copy
from tqdm import tqdm
from termcolor import colored

import sys
sys.path.append("../")

from model.bisenetv2 import BiSeNetV2
from model.cracknet import CrackNet
from model.segnet import SegNet
from model.deeplabv3 import DeepLabv3

from data import Dataset
from config import *
from utils.average_meter import AverageMeter
from utils.util_semseg import SS_Evaluate

class SemSegClient():
    def __init__(self, server_id, client_id, config, clients_dict, tensorboard, dev, time_str):
        self.eid = server_id
        self.cid = client_id
        self.config = config
        self.tb = tensorboard
        self.dev = dev
        self.clients_dict = clients_dict
        self.time_str = time_str

        self.model = self.config['model']().to(self.dev)
        self.model_train     = self.model.train()
        if torch.cuda.is_available():
            self.model_train = torch.nn.DataParallel(self.model_train)
            cudnn.benchmark = True
            self.model_train = self.model_train.cuda()
        self.fedprox_model = copy.deepcopy(self.model)
        self.mu = 0.005 #0: fedavg, float[0,1]: fedprox/feddyn
        self.beta = 1  #0: fedavg/fedprox, 1:feddyn

        self.prev_grads = None
        for param in self.model.parameters():
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = torch.zeros_like(param.view(-1))
            else:
                self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)

        self.batch_size = self.config[self.eid][self.cid]['batch_size']
        self.lr = self.config[self.eid][self.cid]['lr']
        self.betas = self.config[self.eid][self.cid]['betas']
        self.weight_decay = self.config[self.eid][self.cid]['weight_decay']
        self.epochs = self.config['global_round'] * self.config['EAI'] * self.config['CAI']

        self.updated_model = None
        self.epoch_cnt = 0
        #self.criterion = nn.BCELoss().to(self.dev)
        self.criterion = nn.CrossEntropyLoss(ignore_index=19).to(self.dev)

        self.dataset = Dataset(self.config[self.eid][self.cid]['dataset'])
        self.dataloader = DataLoader(self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1, drop_last=True)
        self.test_dataset = Dataset(self.config['test']['dataset'], type_='test')
        self.test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=self.config['test']['batch_size'],
                                     shuffle=True,
                                     num_workers=1, drop_last=True)

        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.lr,
                                      betas=self.betas,
                                      weight_decay=self.weight_decay)

    def train(self):
        for epoch in range(self.config['EAI']):
            if self.updated_model is not None:
                self.model.load_state_dict(self.updated_model)
                self.fedprox_model.load_state_dict(self.updated_model)
                self.updated_model = None

            self.model.train()
            self.model.aux_mode = 'train'

            loss = AverageMeter()
            for imgs, masks, names in tqdm(self.dataloader):
                *logits, = self.model_train(imgs.to(self.dev))
                pred_masks = [F.softmax(logit, dim=1) for logit in logits]
                dist = [self.criterion(pred_mask, masks.to(self.dev)) for pred_mask in pred_masks]
                dist = sum(dist)
                self.optim.zero_grad()

                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.fedprox_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                dist += self.mu / 2 * proximal_term

                lin_penalty = 0.0
                curr_params = None
                for name, param in self.model.named_parameters():
                    if not isinstance(curr_params, torch.Tensor):
                        curr_params = param.view(-1)
                    else:
                        curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

                lin_penalty = torch.sum(curr_params * self.prev_grads)
                dist -= self.beta * lin_penalty

                dist.backward()
                loss.update(dist.item())
                self.optim.step()

                self.prev_grads = None
                for param in self.model.parameters():
                    if not isinstance(self.prev_grads, torch.Tensor):
                        self.prev_grads = param.grad.view(-1).clone()
                    else:
                        self.prev_grads = torch.cat((self.prev_grads, param.grad.view(-1).clone()), dim=0)

            cid = "%s.%s" % (self.eid, self.cid)
            log_info = "[Epoch: %d/%d] [%s.Train.Loss: %f]" % (self.epoch_cnt, self.epochs, cid, loss.avg)
            print(log_info)

            self.tb.add_scalar(cid + '.Train.Loss', loss.avg, self.epoch_cnt)
            '''
            if self.epoch_cnt % 5 == 0:
                avg_iou, avg_pre, avg_recall, avg_f1 = SS_Evaluate(self.model, self.test_dataloader, self.dev)
                self.tb.add_scalar(cid + '.Eval.AvgIOU', avg_iou, self.epoch_cnt)
                self.tb.add_scalar(cid + '.Eval.AvgPrecision', avg_pre, self.epoch_cnt)
                self.tb.add_scalar(cid + '.Eval.AvgRecall', avg_recall, self.epoch_cnt)
                self.tb.add_scalar(cid + '.Eval.AvgF1', avg_f1, self.epoch_cnt)
                log_info = "[Epoch: %d/%d] [%s.Eval.AvgIOU: %f, %s.Eval.AvgPrecision: %f, %s.Eval.AvgRecall: %f, %s.Eval.AvgF1: %f]" % (self.epoch_cnt, self.epochs, cid, avg_iou, cid, avg_pre, cid, avg_recall, cid, avg_f1)
                print(log_info)
            '''

            self.epoch_cnt += 1

            if epoch == self.config['EAI'] - 1:
                self.clients_dict[self.cid] = self.model.state_dict()
