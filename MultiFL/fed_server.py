import copy
import torch
from torch.utils.data import DataLoader
import os
from termcolor import colored
from tensorboardX import SummaryWriter
import datetime

import sys
sys.path.append("../")

from .fed_ss_client import SemSegClient
from .fed_od_client import ObjDectClient
from .fed_cla_client import ClassiClient

from model.bisenetv2 import BiSeNetV2
from model.cracknet import CrackNet
from model.segnet import SegNet
from model.deeplabv3 import DeepLabv3
from model.fasterrcnn.frcnn import FasterRCNN
from model.resnet import ResNet18

from utils.utils import get_classes
from utils.util_semseg import SS_Evaluate, classi_Evaluate, objDect_Evaluate

from data import Dataset, CIFAR10Dataset, FRCNNDataset, frcnn_dataset_collate
from config import *

class EdgeServer():
    def __init__(self, server_id, config, edges_dict, tensorboard, dev, task, time_str):
        self.id = server_id
        self.config = config
        self.clients_num =sum('Agent' in key for key in self.config[self.id])
        self.edges_dict = edges_dict
        self.task = task
        self.tb = tensorboard
        self.dev = dev
        self.logdir = self.config['logdir']
        self.time_str = time_str

        if self.task == 'semSeg':
            self.model = self.config['model']().to(self.dev)
            self.agent = SemSegClient
        elif self.task == 'objDect':
            _, num_classes = get_classes(self.config['classes_path'])
            self.model = self.config['model'](num_classes, self.config['anchors_size']).to(self.dev)
            self.agent= ObjDectClient
        elif self.task == 'classification':
            self.model = self.config['model']().to(self.dev)
            self.agent = ClassiClient

        self.fed_cnt = 0

        self.clients_dict = {}
        self.avgModel = None

        self.clients = []
        for i in range(self.clients_num):
            cid = 'Agent'+str(i)
            self.clients.append(self.agent(self.id, cid, self.config, self.clients_dict, tensorboard, self.dev, self.time_str))
            print('%s.%s has been initialized!' % (self.id, cid))

    def run(self):
        for i in range(self.clients_num):
            self.clients[i].train()

    def FedAgg(self):
        w_avg = {key: 0.0 for key in self.clients_dict['Agent0'].keys()}
        for k in w_avg.keys():
            for i in range(0, len(self.clients_dict)):
                w_avg[k] += self.config[self.id]['Agent' + str(i)]['agg_coef'] * self.clients_dict['Agent' + str(i)][k]
            #w_avg[k] = torch.div(w_avg[k], len(self.clients_dict))

        self.avgModel = copy.deepcopy(w_avg)
        if 'Nonlinear' in self.config:
            for k, v in self.avgModel.items():
                self.avgModel[k] = self.config['Nonlinear'](v)

        self.model.load_state_dict(self.avgModel)

        if self.task == 'semSeg':
            test_dataset = Dataset(self.config['test']['dataset'], type_='test')
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=self.config['test']['batch_size'],
                                         shuffle=True,
                                         num_workers=1)

            #dicts= SS_Evaluate(self.model, test_dataloader, self.dev)
            #for k, v in dicts.items():
            #    if k == 'mIoU':
            #        self.tb.add_scalar('%s.Eval.mIOU' % self.id, v, self.fed_cnt)
            #    elif k == 'mPrecision':
            #        self.tb.add_scalar('%s.Eval.mPrecision' % self.id, v, self.fed_cnt)
            #    elif k == 'mRecall':
            #        self.tb.add_scalar('%s.Eval.mRecall' % self.id, v, self.fed_cnt)
            #    elif k == 'mF1':
            #        self.tb.add_scalar('%s.Eval.mF1' % self.id, v, self.fed_cnt)
            #    else:
            #        self.tb.add_scalar(self.id + '.Eval.' + k + '.IoU', v['IoU'], self.fed_cnt)
            #        self.tb.add_scalar(self.id + '.Eval.' + k + '.Precision', v['Precision'], self.fed_cnt)
            #        self.tb.add_scalar(self.id + '.Eval.' + k + '.Recall', v['Recall'], self.fed_cnt)
            #        self.tb.add_scalar(self.id + '.Eval.' + k + '.F1', v['F1'], self.fed_cnt)

            #log_info = "[Edge FL: %d] [%s.FL.Eval.mIOU: %f, %s.FL.Eval.mPrecision: %f, %s.FL.Eval.mRecall: %f, %s.FL.Eval.mF1: %f]" % (self.fed_cnt, self.id, dicts['mIoU'], self.id, dicts['mPrecision'], self.id, dicts['mRecall'], self.id, dicts['mF1'])
            #print(log_info)
        elif self.task == 'objDect':
            num_val = 0
            val_lines = None
            with open(self.config['test']['val_anno_path'], encoding='utf-8') as f:
                val_lines   = f.readlines()
                num_val   = len(val_lines)
            epoch_step_val = num_val // self.config['test']['batch_size']
            val_dataset = FRCNNDataset(val_lines, self.config['input_shape'], train = False)
            testloader = DataLoader(val_dataset, shuffle = True, batch_size = self.config['test']['batch_size'], num_workers = 4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
            val_loss, ap_dict = objDect_Evaluate(self.model, testloader, self.dev, val_lines, epoch_step_val, self.logdir, self.config['classes_path'], self.fed_cnt, self.config['input_shape'])
            log_info = "[Edge FL: %d] [%s.FL.Eval.Loss: %.3f, %s.FL.Eval.mAP: %.3f%%]" % (self.fed_cnt, self.id, val_loss, self.id, ap_dict['mAP'])
            print(log_info)

            self.tb.add_scalar(self.id + '.FL.Eval.Loss', val_loss, self.fed_cnt)
            for k, v in ap_dict.items():
                if k == 'mAP':
                    self.tb.add_scalar(self.id + '.FL.Eval.mAP', v, self.fed_cnt)
                else:
                    self.tb.add_scalar(self.id + '.FL.Eval.' + k + '.AP', v['AP'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.FL.Eval.' + k + '.F1', v['F1'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.FL.Eval.' + k + '.Recall', v['Recall'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.FL.Eval.' + k + '.Precision', v['Precision'], self.fed_cnt)
        elif self.task == 'classification':
            testset = CIFAR10Dataset(data=self.config['test']['dataset'], mode='test')
            testloader = DataLoader(testset, batch_size=self.config['test']['batch_size'], shuffle=False, num_workers=2)
            val_loss, val_acc = classi_Evaluate(self.model, testloader, self.dev)
            log_info = "[Edge FL: %d] [%s.FL.Eval.Loss: %.3f, %s.FL.Eval.Acc: %.3f%%]" % (self.fed_cnt, self.id, val_loss, self.id, val_acc)
            print(log_info)

            self.tb.add_scalar(self.id + '.FL.Eval.Loss', val_loss, self.fed_cnt)
            self.tb.add_scalar(self.id + '.FL.Eval.Acc', val_acc, self.fed_cnt)

        self.fed_cnt += 1

        self.edges_dict[self.id] = copy.deepcopy(self.avgModel)

    def SinkModelToClients(self):
        for client in self.clients:
            client.updated_model = copy.deepcopy(self.avgModel)

class CloudServer():
    def __init__(self):
        import MultiFL.fed_config as config
        self.task = config.task
        if self.task == 'semSeg':
            self.config = config.semSeg
        elif self.task == 'objDect':
            self.config = config.objDect
        elif self.task == 'classification':
            self.config = config.classi
        else:
            print(colored('Wrong GPU ID!', 'red'))
            exit()

        if self.config['GPU_ID'] >= torch.cuda.device_count():
            print(colored('Wrong GPU ID!', 'red'))
            exit()
        self.dev = torch.device('cuda:%d' % (self.config['GPU_ID']) if torch.cuda.is_available() else 'cpu')

        if self.task == 'semSeg':
            self.model = self.config['model']().to(self.dev)
        elif self.task == 'objDect':
            _, num_classes = get_classes(self.config['classes_path'])
            self.model = self.config['model'](num_classes, self.config['anchors_size']).to(self.dev)
        elif self.task == 'classification':
            self.model = self.config['model']().to(self.dev)

        self.avgModel = copy.deepcopy(self.model.state_dict())

        self.edges_num = sum('Edge' in key for key in self.config)
        self.edges_dict = {}
        self.logdir = self.config['logdir']
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        logruns = 'runs/' if 'Nonlinear' not in self.config else self.config['Nonlinear'].__name__ + '_runs/'
        self.tb = SummaryWriter(self.logdir + '/' + self.task + '/' + self.time_str + '/' + logruns)

        self.fed_cnt = 0

        self.edges = []
        for i in range(self.edges_num):
            eid = 'Edge'+str(i)
            self.edges.append(EdgeServer(eid, self.config, self.edges_dict, self.tb, self.dev, self.task, '/' + self.task + '/' + self.time_str + '/'))

    def run(self):
        save_path = self.logdir + '/' + self.task + '/' + self.time_str + '/' + 'checkpoints/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for j in range(self.config['global_round']):
            for i in range(self.config['CAI']):
                for edge in self.edges:
                    edge.run()

                if i == self.config['CAI'] - 1:
                    self.FedAgg()
                    self.SinkModelToEdges()
                    if j % 10 == 9:
                        save_name = save_path + "/Cloud_FL_%d_model.pth" % j
                        torch.save(self.avgModel, save_name)

                    for edge in self.edges:
                        edge.SinkModelToClients()
                        if j % 10 == 9:
                            save_name = save_path + "/%s_%d_FL_model.pth" % (edge.id, j)
                            torch.save(edge.avgModel, save_name)
                else:
                    for edge in self.edges:
                        edge.FedAgg()
                        edge.SinkModelToClients()
                        if j % 10 == 9:
                            save_name = save_path + "/%s_%d_FL_model.pth" % (edge.id, j)
                            torch.save(edge.avgModel, save_name)

    def FedAgg(self):
        for edge in self.edges:
            edge.FedAgg()

        w_avg = {key: 0.0 for key in self.edges_dict['Edge0'].keys()}
        for k in w_avg.keys():
            for i in range(0, len(self.edges_dict)):
                w_avg[k] += self.config['Edge' + str(i)]['agg_coef'] * self.edges_dict['Edge' + str(i)][k]
            #w_avg[k] = torch.div(w_avg[k], len(self.edges_dict))

        self.avgModel = copy.deepcopy(w_avg)
        if 'Nonlinear' in self.config:
            for k, v in self.avgModel.items():
                self.avgModel[k] = self.config['Nonlinear'](v)

        self.model.load_state_dict(self.avgModel)

        if self.task == 'semSeg':
            test_dataset = Dataset(self.config['test']['dataset'], type_='test')
            test_dataloader = DataLoader(test_dataset,
                                             batch_size=self.config['test']['batch_size'],
                                             shuffle=True,
                                             num_workers=1)

            dicts= SS_Evaluate(self.model, test_dataloader, self.dev)
            for k, v in dicts.items():
                if k == 'mIoU':
                    self.tb.add_scalar('Cloud.Eval.mIOU', v, self.fed_cnt)
                elif k == 'mPrecision':
                    self.tb.add_scalar('Cloud.Eval.mPrecision', v, self.fed_cnt)
                elif k == 'mRecall':
                    self.tb.add_scalar('Cloud.Eval.mRecall', v, self.fed_cnt)
                elif k == 'mF1':
                    self.tb.add_scalar('Cloud.Eval.mF1', v, self.fed_cnt)
                else:
                    self.tb.add_scalar('Cloud.Eval.' + k + '.IoU', v['IoU'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.' + k + '.Precision', v['Precision'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.' + k + '.Recall', v['Recall'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.' + k + '.F1', v['F1'], self.fed_cnt)

            log_info = "[Cloud FL: %d] [Cloud.FL.Eval.AvgIOU: %f, Cloud.FL.Eval.AvgPrecision: %f, Cloud.FL.Eval.AvgRecall: %f, Cloud.FL.Eval.AvgF1: %f]" % (self.fed_cnt, dicts['mIoU'], dicts['mPrecision'], dicts['mRecall'], dicts['mF1'])
            print(log_info)
        elif self.task == 'objDect':
            num_val = 0
            val_lines = None
            with open(self.config['test']['val_anno_path'], encoding='utf-8') as f:
                val_lines   = f.readlines()
                num_val   = len(val_lines)
            epoch_step_val = num_val // self.config['test']['batch_size']
            val_dataset = FRCNNDataset(val_lines, self.config['input_shape'], train = False)
            testloader = DataLoader(val_dataset, shuffle = True, batch_size = self.config['test']['batch_size'], num_workers = 4, pin_memory=True, drop_last=True, collate_fn=frcnn_dataset_collate)
            val_loss, ap_dict = objDect_Evaluate(self.model, testloader, self.dev, val_lines, epoch_step_val, self.logdir, self.config['classes_path'], self.fed_cnt, self.config['input_shape'])
            log_info = "[Cloud FL: %d] [FL.Eval.Loss: %.3f, FL.Eval.mAP: %.3f%%]" % (self.fed_cnt, val_loss, ap_dict['mAP'])
            print(log_info)

            self.tb.add_scalar('Cloud.FL.Eval.Loss', val_loss, self.fed_cnt)
            for k, v in ap_dict.items():
                if k == 'mAP':
                    self.tb.add_scalar('Cloud.FL.Eval' + '.mAP', v, self.fed_cnt)
                else:
                    self.tb.add_scalar('Cloud.FL.Eval' + '.' + k + '.AP', v['AP'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.FL.Eval' + '.' + k + '.F1', v['F1'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.FL.Eval' + '.' + k + '.Recall', v['Recall'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.FL.Eval' + '.' + k + '.Precision', v['Precision'], self.fed_cnt)
        elif self.task == 'classification':
            testset = CIFAR10Dataset(data=self.config['test']['dataset'], mode='test')
            testloader = DataLoader(testset, batch_size=self.config['test']['batch_size'], shuffle=False, num_workers=2)
            val_loss, val_acc = classi_Evaluate(self.model, testloader, self.dev)
            log_info = "[Cloud FL: %d] [FL.Eval.Loss: %.3f, FL.Eval.Acc: %.3f%%]" % (self.fed_cnt, val_loss, val_acc)
            print(log_info)

            self.tb.add_scalar('Cloud.FL.Eval.Loss', val_loss, self.fed_cnt)
            self.tb.add_scalar('Cloud.FL.Eval.Acc', val_acc, self.fed_cnt)

        self.fed_cnt += 1

    def SinkModelToEdges(self):
        for edge in self.edges:
            edge.avgModel = copy.deepcopy(self.avgModel)
