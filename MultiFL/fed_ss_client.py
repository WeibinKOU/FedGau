import numpy as np
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import copy
import math
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
from utils.moon_criterion import MOONLoss

class SemSegClient():
    def __init__(self, server_id, client_id, config, clients_dict, tensorboard, dev, time_str):
        self.eid = server_id
        self.cid = client_id
        self.config = config
        self.tb = tensorboard
        self.dev = dev
        self.clients_dict = clients_dict
        self.time_str = time_str

        if self.config['dataset'] == 'cityscapes':
            self.dataset = Dataset(self.config[self.eid][self.cid]['dataset'], num_classes=20)
            self.test_dataset = Dataset(self.config['test']['dataset'], num_classes=20, type_='test')
            self.criterion = nn.CrossEntropyLoss(ignore_index=19).to(self.dev)
            self.model = self.config['model'](n_classes=20).to(self.dev)
            self.n_classes = 20
        elif self.config['dataset'] == 'Mapillary':
            self.dataset = Dataset(self.config[self.eid][self.cid]['dataset'], num_classes=66)
            self.test_dataset = Dataset(self.config['test']['dataset'], num_classes=66, type_='test')
            self.criterion = nn.CrossEntropyLoss(ignore_index=65).to(self.dev)
            self.model = self.config['model'](n_classes=66).to(self.dev)
            self.n_classes = 66
        elif self.config['dataset'] == 'CamVid':
            self.dataset = Dataset(self.config[self.eid][self.cid]['dataset'], num_classes=12)
            self.test_dataset = Dataset(self.config['test']['dataset'], num_classes=12, type_='test')
            self.criterion = nn.CrossEntropyLoss(ignore_index=11).to(self.dev)
            self.model = self.config['model'](n_classes=12).to(self.dev)
            self.n_classes = 12
        else:
            print('Dataset %s is not supported!'%self.config['dataset'])
            exit()

        self.model_train     = self.model.train()
        if torch.cuda.is_available():
            self.model_train = torch.nn.DataParallel(self.model_train)
            cudnn.benchmark = True
            self.model_train = self.model_train.cuda()
        self.fedprox_model = copy.deepcopy(self.model)

        if 'FedAlgo' not in self.config:
            self.mu = 0.0
            self.alpha = 0.0
        elif self.config['FedAlgo'] == 'FedAvg' or self.config['FedAlgo'] == 'FedStats' or 'FedAvgM' in self.config['FedAlgo'] or self.config['FedAlgo'] == 'FedIR' or 'FedCurv' in self.config['FedAlgo'] or self.config['FedAlgo'] == 'MOON' or self.config['FedAlgo'] == 'SCAFFOLD':
            self.mu = 0.0
            self.alpha = 0.0
        elif 'FedProx' in self.config['FedAlgo']:
            self.mu = float(self.config['FedAlgo'].split('-')[-1])
            self.alpha = 0.0
        elif 'FedDyn' in self.config['FedAlgo']:
            self.mu = float(self.config['FedAlgo'].split('-')[-1])
            self.alpha = 1.0

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

        self.epoch_cnt = 0
        self.eval_loss = 0.0
        self.delta_ce = 0.0
        self.rho = 0.0
        self.beta = 0.0
        self.grad = None

        self.dataloader = DataLoader(self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1, drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=self.config['test']['batch_size'],
                                     shuffle=True,
                                     num_workers=1, drop_last=True)

        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=self.lr,
                                      betas=self.betas,
                                      weight_decay=self.weight_decay)

        if self.config['FedAlgo'] == 'FedIR':
            total = 0
            counts = torch.zeros(self.n_classes)
            for _, masks, _ in tqdm(self.dataloader):
                masks = masks.view(-1)
                total += masks.size(0)
                counts += torch.bincount(masks, minlength=self.n_classes)

            weights = counts / total
            self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=self.n_classes - 1).to(self.dev)

        if 'FedCurv' in self.config['FedAlgo']:
            self.fisher_lambda = float(self.config['FedAlgo'].split('-')[-1])
            self.Pt = None
            self.Qt = None
            self.is_FedCurv_running = False

        if 'MOON' == self.config['FedAlgo']:
            self.criterion = MOONLoss(ign_idx=self.n_classes - 1).to(self.dev)
            self.dg_model = copy.deepcopy(self.model)
            self.prev_model = copy.deepcopy(self.model)

        if 'SCAFFOLD' == self.config['FedAlgo']:
            self.scfd_c = torch.from_numpy(self.flatten_weights(self.model)).fill_(0)
            self.scfd_c_i = self.scfd_c
            self.scfd_ud_amt = None
            self.scfd_step_cnt = 0
            self.scfd_rho = self.optim.param_groups[0]['betas'][0]
            self.scfd_lr = self.optim.param_groups[0]['lr']
            self.adaptive_divison = True

    def train(self):
        for epoch in range(self.config['EAI']):
            if self.updated_model is not None:
                self.model.load_state_dict(self.updated_model)
                self.fedprox_model.load_state_dict(self.updated_model)
                self.updated_model = None

                if 'MOON' == self.config['FedAlgo'] or 'SCAFFOLD' == self.config['FedAlgo']:
                    self.dg_model = copy.deepcopy(self.model)
                    self.scfd_step_cnt = 0

                self.prev_grads = None
                for param in self.model.parameters():
                    if not isinstance(self.prev_grads, torch.Tensor):
                        self.prev_grads = torch.zeros_like(param.view(-1))
                    else:
                        self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)

            self.model.train()
            self.model.aux_mode = 'train'

            loss = AverageMeter()
            diff_term = 0.0

            for imgs, masks, names in tqdm(self.dataloader):
                if 'MOON' == self.config['FedAlgo']:
                    *logits, z = self.model_train(imgs.to(self.dev), get_feats=True)
                    *logits_prev, z_prev = self.prev_model(imgs.to(self.dev), get_feats=True)
                    *logits_g, z_g = self.dg_model(imgs.to(self.dev), get_feats=True)
                    dist = self.criterion(logits, masks.to(self.dev), z, z_prev, z_g)
                else:
                    *logits, = self.model_train(imgs.to(self.dev))
                    pred_masks = [F.softmax(logit, dim=1) for logit in logits]
                    dist = [self.criterion(pred_mask, masks.to(self.dev)) for pred_mask in pred_masks]
                    dist = sum(dist)
                self.optim.zero_grad()

                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.fedprox_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                dist += self.mu / 2 * proximal_term
                diff_term = proximal_term.item()

                lin_penalty = 0.0
                curr_params = None
                for name, param in self.model.named_parameters():
                    if not isinstance(curr_params, torch.Tensor):
                        curr_params = param.view(-1)
                    else:
                        curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

                lin_penalty = torch.sum(curr_params * self.prev_grads)
                dist -= self.alpha * lin_penalty

                if 'FedCurv' in self.config['FedAlgo'] and self.is_FedCurv_running:
                    all_params = self.flatten_weights(self.model, numpy_output=False)
                    reg_loss = self.fisher_lambda * torch.inner(self.Pt, torch.square(all_params)) - self.fisher_lambda * 2.0 * torch.inner(self.Qt, all_params)
                    dist += reg_loss

                dist.backward()
                loss.update(dist.item())

                if 'SCAFFOLD' == self.config['FedAlgo']:
                    self.scfd_c_i = self.scfd_c_i.to(self.dev)
                    self.scfd_c = self.scfd_c.to(self.dev)
                    grad_batch = self.flatten_grads(self.model).detach().clone()
                    self.optim.zero_grad()
                    grad_batch = grad_batch - self.scfd_c_i + self.scfd_c
                    self.model = self.assign_grads(self.model, grad_batch)
                    self.scfd_step_cnt += 1

                self.optim.step()

                self.prev_grads = None
                for param in self.model.parameters():
                    if not isinstance(self.prev_grads, torch.Tensor):
                        self.prev_grads = param.grad.view(-1).clone()
                    else:
                        self.prev_grads = torch.cat((self.prev_grads, param.grad.view(-1).clone()), dim=0)
                self.grad = self.prev_grads


            cid = "%s.%s" % (self.eid, self.cid)
            log_info = "[Epoch: %d/%d] [%s.Train.Loss: %f]" % (self.epoch_cnt, self.epochs, cid, loss.avg)
            print(log_info)
            self.tb.add_scalar(cid + '.Train.Loss', loss.avg, self.epoch_cnt)
            self.epoch_cnt += 1

            if epoch == self.config['EAI'] - 1:
                self.clients_dict[self.cid] = self.model.state_dict()
                if 'enable_optim' in self.config and self.config['enable_optim']:
                    self.model.eval()
                    self.fedprox_model.eval()
                    _, _, self.eval_loss = SS_Evaluate(self.model, self.test_dataloader, self.dev, self.config['dataset'])
                    _, _, eval_loss = SS_Evaluate(self.fedprox_model, self.test_dataloader, self.dev, self.config['dataset'])
                    self.rho = abs(self.eval_loss - eval_loss) / (math.sqrt(diff_term) + 1e-6)
                    self.rho = self.rho.item()

                    grad_diff = None
                    for param_c, param_e in zip(self.model.parameters(), self.fedprox_model.parameters()):
                        if not isinstance(grad_diff, torch.Tensor):
                            grad_diff = (param_c.grad - param_e.grad).view(-1).clone()
                        else:
                            grad_diff = torch.cat((grad_diff, (param_c.grad - param_e.grad).view(-1).clone()), dim=0)
                    self.delta_ce = math.sqrt(grad_diff.norm(2).item())
                    self.beta = self.delta_ce / (math.sqrt(diff_term) + 1e-6)
                    del grad_diff

                if 'FedCurv' in self.config['FedAlgo']:
                    self.calc_local_fisher()


                if 'MOON' == self.config['FedAlgo']:
                    self.prev_model = copy.deepcopy(self.model)

                if 'SCAFFOLD' == self.config['FedAlgo']:
                    self.update_control_variate()
                    #print('step counts: ', self.scfd_step_cnt)
                    #print('c_i: ', self.scfd_c_i)
                    #print('c_amount: ', self.scfd_ud_amt)
                    self.scfd_step_cnt = 0

    def flatten_weights(self, model, numpy_output=True):
        all_params = []
        for param in model.parameters():
            all_params.append(param.view(-1))

        all_params = torch.cat(all_params)
        if numpy_output:
            return all_params.cpu().detach().numpy()

        return all_params

    def calc_local_fisher(self):
        local_params = self.flatten_weights(self.model, numpy_output=False).clone().detach()
        fisher_list = []

        for imgs, masks, _ in tqdm(self.dataloader):
            batch_size = len(masks)

            data, targets = imgs.to(self.dev), masks.to(self.dev)

            *logits, = self.model_train(imgs.to(self.dev))
            pred_masks = [F.softmax(logit, dim=1) for logit in logits]
            crit = [self.criterion(pred_mask, masks.to(self.dev)) for pred_mask in pred_masks]
            crit = sum(crit)

            grad = autograd.grad(crit, self.model.parameters())
            all_grad_eles = []
            for elewise in grad:
                all_grad_eles.append(elewise.view(-1))
            all_grad_eles = torch.cat(all_grad_eles)
            fisher_list.append(torch.square(all_grad_eles.clone().detach()))

        self.Pt = torch.mean(torch.stack(fisher_list), dim=0).clone().detach()
        self.Qt = torch.mul(self.Pt, local_params).clone().detach()

    def flatten_grads(self, model):
        all_grads = []
        for name, param in model.named_parameters():
            all_grads.append(param.grad.view(-1))
        return torch.cat(all_grads)

    def assign_grads(self, model, grads):
        state_dict = model.state_dict(keep_vars=True)
        index = 0
        for param in state_dict.keys():
            # ignore batchnorm params
            if ("running_mean" in param or "running_var" in param or "num_batches_tracked" in param):
                continue

            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param].grad = grads[index : index + param_count].view(param_shape).clone()
            index += param_count
        model.load_state_dict(state_dict)
        return model

    def get_divisor(self):
        local_lr = self.scfd_lr
        K = self.scfd_step_cnt
        rho = self.scfd_rho

        new_K = (K - rho * (1.0 - pow(rho, K)) / (1.0 - rho)) / (1.0 - rho)

        if self.adaptive_divison:
            divisor = 1.0 / (new_K * local_lr)
        else:
            divisor = 1.0 / (K * local_lr)

        return divisor

    @torch.no_grad()
    def update_control_variate(self):
        divisor = self.get_divisor()

        server_params = self.flatten_weights(self.dg_model)
        local_params = self.flatten_weights(self.model)
        param_move = server_params - local_params

        c_i_plus = self.scfd_c_i.cpu() - self.scfd_c.cpu() + divisor * param_move
        c_update_amount = c_i_plus - self.scfd_c_i.cpu()

        self.scfd_c_i = c_i_plus
        self.scfd_ud_amt = c_update_amount
