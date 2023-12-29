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
from .fed_adaptive_optimizer import adaptiveOptimizer

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
        self.eval_loss = 0.0
        self.eval_loss_diff = 0.0
        self.delta_e = 0.0
        self.rho = 0.0
        self.beta = 0.0
        self.grad = None

        if self.task == 'semSeg':
            if self.config['dataset'] == 'cityscapes':
                self.model = self.config['model'](n_classes=20).to(self.dev)
            elif self.config['dataset'] == 'Mapillary':
                self.model = self.config['model'](n_classes=66).to(self.dev)
            elif self.config['dataset'] == 'CamVid':
                self.model = self.config['model'](n_classes=12).to(self.dev)
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
            if self.config['dataset'] == 'cityscapes':
                test_dataset = Dataset(self.config['test']['dataset'], num_classes=20, type_='test')
            elif self.config['dataset'] == 'Mapillary':
                test_dataset = Dataset(self.config['test']['dataset'], num_classes=66, type_='test')
            elif self.config['dataset'] == 'CamVid':
                test_dataset = Dataset(self.config['test']['dataset'], num_classes=12, type_='test')
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=self.config['test']['batch_size'],
                                         shuffle=True,
                                         num_workers=1,
                                         drop_last=True)

            clsdicts, catdicts, self.eval_loss = SS_Evaluate(self.model, test_dataloader, self.dev, self.config['dataset'])
            self.tb.add_scalar('%s.Eval.Loss' % self.id, self.eval_loss, self.fed_cnt)

            for k, v in clsdicts.items():
                if k == 'mIoU':
                    self.tb.add_scalar('%s.Eval.Class.mIoU' % self.id, 100*v, self.fed_cnt)
                elif k == 'mPrecision':
                    self.tb.add_scalar('%s.Eval.Class.mPrecision' % self.id, 100*v, self.fed_cnt)
                elif k == 'mRecall':
                    self.tb.add_scalar('%s.Eval.Class.mRecall' % self.id, 100*v, self.fed_cnt)
                elif k == 'mF1':
                    self.tb.add_scalar('%s.Eval.Class.mF1' % self.id, 100*v, self.fed_cnt)
                else:
                    self.tb.add_scalar(self.id + '.Eval.Class.' + k + '.IoU', 100*v['IoU'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.Eval.Class.' + k + '.Precision', 100*v['Precision'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.Eval.Class.' + k + '.Recall', 100*v['Recall'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.Eval.Class.' + k + '.F1', 100*v['F1'], self.fed_cnt)

            for k, v in catdicts.items():
                if k == 'mIoU':
                    self.tb.add_scalar('%s.Eval.Category.mIoU', 100*v, self.fed_cnt)
                elif k == 'mPrecision':
                    self.tb.add_scalar('%s.Eval.Category.mPrecision', 100*v, self.fed_cnt)
                elif k == 'mRecall':
                    self.tb.add_scalar('%s.Eval.Category.mRecall', 100*v, self.fed_cnt)
                elif k == 'mF1':
                    self.tb.add_scalar('%s.Eval.Category.mF1', 100*v, self.fed_cnt)
                else:
                    self.tb.add_scalar(self.id + '.Eval.Category.' + k + '.IoU', 100*v['IoU'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.Eval.Category.' + k + '.Precision', 100*v['Precision'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.Eval.Category.' + k + '.Recall', 100*v['Recall'], self.fed_cnt)
                    self.tb.add_scalar(self.id + '.Eval.Category.' + k + '.F1', 100*v['F1'], self.fed_cnt)

            log_cls_info = "[Edge FL: %d] [%s.FL.Eval.Class.mIoU: %.2f%%, %s.FL.Eval.Class.mPrecision: %.2f%%, %s.FL.Eval.Class.mRecall: %.2f%%, %s.FL.Eval.Class.mF1: %.2f%%]" % (self.fed_cnt, self.id, 100*clsdicts['mIoU'], self.id, 100*clsdicts['mPrecision'], self.id, 100*clsdicts['mRecall'], self.id, 100*clsdicts['mF1'])
            log_cat_info = "[Edge FL: %d] [%s.FL.Eval.Category.mIoU: %.2f%%, %s.FL.Eval.Category.mPrecision: %.2f%%, %s.FL.Eval.Category.mRecall: %.2f%%, %s.FL.Eval.Category.mF1: %.2f%%]" % (self.fed_cnt, self.id, 100*catdicts['mIoU'], self.id, 100*catdicts['mPrecision'], self.id, 100*catdicts['mRecall'], self.id, 100*catdicts['mF1'])
            print(log_cls_info)
            print(log_cat_info)

            aggted_loss = sum([self.config[self.id]['Agent' + str(i)]['agg_coef'] * self.clients[i].eval_loss for i in range(len(self.clients))])
            self.delta_e = sum([self.config[self.id]['Agent' + str(i)]['agg_coef'] * self.clients[i].delta_ce for i in range(len(self.clients))])
            self.rho = sum([self.config[self.id]['Agent' + str(i)]['agg_coef'] * self.clients[i].rho for i in range(len(self.clients))])
            self.beta = sum([self.config[self.id]['Agent' + str(i)]['agg_coef'] * self.clients[i].beta for i in range(len(self.clients))])
            self.grad = sum([self.config[self.id]['Agent' + str(i)]['agg_coef'] * self.clients[i].grad for i in range(len(self.clients))], torch.zeros_like(self.clients[0].grad))
            self.eval_loss_diff = self.eval_loss - aggted_loss

            self.tb.add_scalar(self.id + '.Optim.AggregatedLoss', aggted_loss, self.fed_cnt)
            self.tb.add_scalar(self.id + '.Optim.GradDelta', self.delta_e, self.fed_cnt)
            self.tb.add_scalar(self.id + '.Optim.Rho', self.rho, self.fed_cnt)
            self.tb.add_scalar(self.id + '.Optim.Beta', self.beta, self.fed_cnt)
            self.tb.add_scalar(self.id + '.Optim.GradNorm', np.sqrt(self.grad.norm(2).item()), self.fed_cnt)
            self.tb.add_scalar(self.id + '.Optim.LossDiff', self.eval_loss_diff, self.fed_cnt)
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
            if self.config['dataset'] == 'cityscapes':
                self.model = self.config['model'](n_classes=20).to(self.dev)
            elif self.config['dataset'] == 'Mapillary':
                self.model = self.config['model'](n_classes=66).to(self.dev)
            elif self.config['dataset'] == 'CamVid':
                self.model = self.config['model'](n_classes=12).to(self.dev)
            else:
                print('Dataset %s is not supported!'%self.config['dataset'])
                exit()
            print('Dataset %s is used!'%self.config['dataset'])
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
        self.hetero_ce = 0.0
        self.hetero_eg = 0.0
        self.delta = 0.0
        self.rho = 0.0
        self.beta = 0.0
        self.grad = None
        self.optimizer = adaptiveOptimizer(self.edges_num, self.config)

        self.edges = []
        for i in range(self.edges_num):
            eid = 'Edge'+str(i)
            self.edges.append(EdgeServer(eid, self.config, self.edges_dict, self.tb, self.dev, self.task, '/' + self.task + '/' + self.time_str + '/'))
        self.clients_num = sum([len(edge.clients) for edge in self.edges])
        self.cq = []
        self.curr_cq = 0.0
        self.last_perf = 0.0
        self.curr_perf = 0.0
        self.traffic = 0

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
                            #torch.save(edge.avgModel, save_name)
                else:
                    for edge in self.edges:
                        edge.FedAgg()
                        edge.SinkModelToClients()
                        if j % 10 == 9:
                            save_name = save_path + "/%s_%d_FL_model.pth" % (edge.id, j)
                            #torch.save(edge.avgModel, save_name)

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
            if self.config['dataset'] == 'cityscapes':
                test_dataset = Dataset(self.config['test']['dataset'], num_classes=20, type_='test')
            elif self.config['dataset'] == 'Mapillary':
                test_dataset = Dataset(self.config['test']['dataset'], num_classes=66, type_='test')
            elif self.config['dataset'] == 'CamVid':
                test_dataset = Dataset(self.config['test']['dataset'], num_classes=12, type_='test')

            test_dataloader = DataLoader(test_dataset,
                                         batch_size=self.config['test']['batch_size'],
                                         shuffle=True,
                                         num_workers=1,
                                         drop_last=True)

            clsdicts, catdicts, eval_loss = SS_Evaluate(self.model, test_dataloader, self.dev, self.config['dataset'])
            self.tb.add_scalar('Cloud.Eval.Loss', eval_loss, self.fed_cnt)
            for k, v in clsdicts.items():
                if k == 'mIoU':
                    self.tb.add_scalar('Cloud.Eval.Class.mIoU', 100*v, self.fed_cnt)
                elif k == 'mPrecision':
                    self.tb.add_scalar('Cloud.Eval.Class.mPrecision', 100*v, self.fed_cnt)
                elif k == 'mRecall':
                    self.tb.add_scalar('Cloud.Eval.Class.mRecall', 100*v, self.fed_cnt)
                elif k == 'mF1':
                    self.tb.add_scalar('Cloud.Eval.Class.mF1', 100*v, self.fed_cnt)
                else:
                    self.tb.add_scalar('Cloud.Eval.Class.' + k + '.IoU', 100*v['IoU'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.Class.' + k + '.Precision', 100*v['Precision'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.Class.' + k + '.Recall', 100*v['Recall'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.Class.' + k + '.F1', 100*v['F1'], self.fed_cnt)

            for k, v in catdicts.items():
                if k == 'mIoU':
                    self.curr_perf = 100*v
                    self.tb.add_scalar('Cloud.Eval.Category.mIoU', 100*v, self.fed_cnt)
                elif k == 'mPrecision':
                    self.tb.add_scalar('Cloud.Eval.Category.mPrecision', 100*v, self.fed_cnt)
                elif k == 'mRecall':
                    self.tb.add_scalar('Cloud.Eval.Category.mRecall', 100*v, self.fed_cnt)
                elif k == 'mF1':
                    self.tb.add_scalar('Cloud.Eval.Category.mF1', 100*v, self.fed_cnt)
                else:
                    self.tb.add_scalar('Cloud.Eval.Category.' + k + '.IoU', 100*v['IoU'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.Category.' + k + '.Precision', 100*v['Precision'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.Category.' + k + '.Recall', 100*v['Recall'], self.fed_cnt)
                    self.tb.add_scalar('Cloud.Eval.Category.' + k + '.F1', 100*v['F1'], self.fed_cnt)

            log_cls_info = "[Cloud FL: %d] [Cloud.FL.Eval.Class.mIoU: %.2f%%, Cloud.FL.Eval.Class.mPrecision: %.2f%%, Cloud.FL.Eval.Class.mRecall: %.2f%%, Cloud.FL.Eval.Class.mF1: %.2f%%]" % (self.fed_cnt, 100*clsdicts['mIoU'], 100*clsdicts['mPrecision'], 100*clsdicts['mRecall'], 100*clsdicts['mF1'])
            log_cat_info = "[Cloud FL: %d] [Cloud.FL.Eval.Category.mIoU: %.2f%%, Cloud.FL.Eval.Category.mPrecision: %.2f%%, Cloud.FL.Eval.Category.mRecall: %.2f%%, Cloud.FL.Eval.Category.mF1: %.2f%%]" % (self.fed_cnt, 100*catdicts['mIoU'], 100*catdicts['mPrecision'], 100*catdicts['mRecall'], 100*catdicts['mF1'])
            print(log_cls_info)
            print(log_cat_info)

            traffic = 2 * (self.config['CAI'] * sum([len(edge.clients) for edge in self.edges]) + len(self.edges))
            self.traffic += traffic
            self.tb.add_scalar('Cloud.Traffic', self.traffic, self.fed_cnt)
            if self.last_perf > 0:
                self.curr_cq = (self.curr_perf - self.last_perf) / (traffic + 1e-6) if self.curr_perf - self.last_perf > 0 else 0
                self.cq.append(self.curr_cq)

            self.hetero_eg = eval_loss - sum(self.config['Edge' + str(i)]['agg_coef'] * self.edges[i].eval_loss for i in range(len(self.edges)))
            self.hetero_eg = self.hetero_eg.item()
            self.hetero_ce = sum(self.config['Edge' + str(i)]['agg_coef'] * self.edges[i].eval_loss_diff for i in range(len(self.edges)))
            self.hetero_ce = self.hetero_ce.item()
            self.delta = sum(self.config['Edge' + str(i)]['agg_coef'] * self.edges[i].delta_e for i in range(len(self.edges)))
            self.rho = sum(self.config['Edge' + str(i)]['agg_coef'] * self.edges[i].rho for i in range(len(self.edges)))
            self.beta = sum(self.config['Edge' + str(i)]['agg_coef'] * self.edges[i].beta for i in range(len(self.edges)))
            self.grad = sum([self.config['Edge' + str(i)]['agg_coef'] * self.edges[i].grad for i in range(len(self.edges))], torch.zeros_like(self.edges[0].grad))
            self.grad = self.grad.norm(2).item()
            optim_theta = self.curr_cq / (max(self.cq) + 1e-6) if self.last_perf > 0 else 1.0


            self.tb.add_scalar('Cloud.Optim.Hetero_EG', self.hetero_eg, self.fed_cnt)
            self.tb.add_scalar('Cloud.Optim.Hetero_CE', self.hetero_ce, self.fed_cnt)
            self.tb.add_scalar('Cloud.Optim.AdaptiveFactor', optim_theta, self.fed_cnt)
            self.tb.add_scalar('Cloud.Optim.GradDelta', self.delta, self.fed_cnt)
            self.tb.add_scalar('Cloud.Optim.GradNorm', np.sqrt(self.grad), self.fed_cnt)
            self.tb.add_scalar('Cloud.Optim.Rho', self.rho, self.fed_cnt)
            self.tb.add_scalar('Cloud.Optim.Beta', self.beta, self.fed_cnt)

            self.optimizer.theta = optim_theta
            self.optimizer.beta = self.beta
            self.optimizer.rho = self.rho
            self.optimizer.delta = self.delta
            self.optimizer.delta_e.clear()
            for i in range(len(self.edges)):
                self.optimizer.delta_e.append(self.edges[i].delta_e)
            self.optimizer.grad_norm2.append(self.grad)
            estiC = 0.0
            if 'enable_optim' in self.config and self.config['enable_optim']:
                eai, cai, estiC = self.optimizer.solve()
                print('Optimized Result (EAI, CAI):', eai, cai)
                self.config['EAI'] = eai
                self.config['CAI'] = cai
                for edge in self.edges:
                    edge.config['EAI'] = eai
                    edge.config['CAI'] = cai
                    for client in edge.clients:
                        client.config['EAI'] = eai
                        client.config['CAI'] = cai

            self.tb.add_scalar('Cloud.Optim.EstimC', estiC, self.fed_cnt)
            self.tb.add_scalar('Cloud.Optim.EAI', self.config['EAI'], self.fed_cnt)
            self.tb.add_scalar('Cloud.Optim.CAI', self.config['CAI'], self.fed_cnt)

            self.last_perf = self.curr_perf
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
