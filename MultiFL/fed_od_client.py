#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler as GradScaler

import sys
sys.path.append("../")

from model.fasterrcnn.frcnn import FasterRCNN
from model.fasterrcnn.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from data import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

class ObjDectClient():
    def __init__(self, server_id, client_id, config, clients_dict, tensorboard, dev, time_str):
        self.eid = server_id
        self.cid = client_id
        self.config = config
        self.tb = tensorboard
        self.dev = dev 
        self.time_str = time_str

        self.cuda            = torch.cuda.is_available() 
        self.train_gpu       = list(range(torch.cuda.device_count()))
        self.fp16            = True 

        self.train_anno_path = self.config[self.eid][self.cid]['train_anno_path']
        self.batch_size      = self.config[self.eid][self.cid]['batch_size']
        self.Init_lr         = self.config[self.eid][self.cid]['lr']
        self.betas           = self.config[self.eid][self.cid]['betas']
        self.weight_decay    = self.config[self.eid][self.cid]['weight_decay']

        self.epochs          = self.config['global_round'] * self.config['EAI'] * self.config['CAI'] 
        self.classes_path    = self.config['classes_path']
        self.val_anno_path   = self.config['test']['val_anno_path']
        self.save_dir        = self.config['logdir']
        self.lr_decay_type   = self.config['lr_decay_type']
        self.save_period     = self.config['save_period']
        self.eval_period     = self.config['eval_period']
        self.input_shape     = self.config['input_shape']
        self.anchors_size    = self.config['anchors_size']

        self.clients_dict    = clients_dict
        self.Min_lr          = self.Init_lr * 0.01
        self.num_workers     = 4
        self.eval_flag       = True
        self.epoch_cnt       = 0
        self.updated_model = None

        self.class_names, self.num_classes = get_classes(self.classes_path)

        os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in self.train_gpu)
    
        self.model = FasterRCNN(self.num_classes, anchor_scales = self.anchors_size)

        self.log_dir = self.save_dir + self.time_str
        self.loss_history    = LossHistory(self.log_dir, self.model, input_shape=self.input_shape)
        self.scaler = GradScaler()
        
        self.model_train     = self.model.train()
        if self.cuda:
            self.model_train = torch.nn.DataParallel(self.model_train)
            cudnn.benchmark = True
            self.model_train = self.model_train.cuda()

        with open(self.train_anno_path, encoding='utf-8') as f:
            self.train_lines = f.readlines()
        with open(self.val_anno_path, encoding='utf-8') as f:
            self.val_lines   = f.readlines()
        self.num_train   = len(self.train_lines)
        self.num_val     = len(self.val_lines)

        weights_init(self.model)

        self.nbs             = 16
        self.lr_limit_max    = 1e-4
        self.lr_limit_min    = 1e-4
        self.Init_lr_fit     = min(max(self.batch_size / self.nbs * self.Init_lr, self.lr_limit_min), self.lr_limit_max)
        self.Min_lr_fit      = min(max(self.batch_size / self.nbs * self.Min_lr, self.lr_limit_min * 1e-2), self.lr_limit_max * 1e-2)

        self.optimizer = optim.Adam(self.model.parameters(), self.Init_lr_fit, betas = self.betas, weight_decay = self.weight_decay)

        self.lr_scheduler_func = get_lr_scheduler(self.lr_decay_type, self.Init_lr_fit, self.Min_lr_fit, self.epochs)

        self.epoch_step      = self.num_train // self.batch_size
        self.epoch_step_val  = self.num_val // self.batch_size

        self.train_dataset   = FRCNNDataset(self.train_lines, self.input_shape, train = True)
        self.val_dataset     = FRCNNDataset(self.val_lines, self.input_shape, train = False)

        self.gen             = DataLoader(self.train_dataset, shuffle = True, batch_size = self.batch_size, num_workers = self.num_workers, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
        self.gen_val         = DataLoader(self.val_dataset  , shuffle = True, batch_size = self.batch_size, num_workers = self.num_workers, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)

        self.train_util      = FasterRCNNTrainer(self.model_train, self.optimizer)
        self.eval_callback   = EvalCallback(self.model_train, self.input_shape, self.class_names, self.num_classes, self.val_lines, self.log_dir, self.cuda, eval_flag=self.eval_flag, period=self.eval_period)


    def train(self):
        if self.updated_model is not None:
            self.model.load_state_dict(self.updated_model)
            self.updated_model = None

        self.model.freeze_bn()

        for epoch in range(self.config['EAI']):
            set_optimizer_lr(self.optimizer, self.lr_scheduler_func, self.epoch_cnt)
            
            train_loss, val_loss, ap_dict = fit_one_epoch(self.model, self.train_util, self.loss_history, 
                                                          self.eval_callback, self.optimizer, self.epoch_cnt,
                                                          self.epoch_step, self.epoch_step_val, self.gen, self.gen_val,
                                                          self.epochs, self.cuda, self.fp16, self.scaler,
                                                          self.save_period, self.save_dir)

            cid = "%s.%s" % (self.eid, self.cid)
            log_info = "[%s][Epoch: %d/%d][Train.Loss: %f, Valid.Loss: %f]" % (cid, self.epoch_cnt, self.epochs, train_loss, val_loss)
            print(log_info)

            self.tb.add_scalar(cid + '.Train.Loss', train_loss, self.epoch_cnt)
            self.tb.add_scalar(cid + '.Valid.Loss', val_loss, self.epoch_cnt)

            for k, v in ap_dict.items():
                if k == 'mAP':
                    self.tb.add_scalar(cid + '.mAP', v, self.epoch_cnt)
                else:
                    self.tb.add_scalar(cid + '.' + k + '.AP', v['AP'], self.epoch_cnt)
                    self.tb.add_scalar(cid + '.' + k + '.F1', v['F1'], self.epoch_cnt)
                    self.tb.add_scalar(cid + '.' + k + '.Recall', v['Recall'], self.epoch_cnt)
                    self.tb.add_scalar(cid + '.' + k + '.Precision', v['Precision'], self.epoch_cnt)

            self.epoch_cnt += 1

            if epoch == self.config['EAI'] - 1:
                self.clients_dict[self.cid] = self.model.state_dict()
