import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import argparse
from termcolor import colored

import sys
sys.path.append("../")

from model.resnet import ResNet18
from utils.utils_cla import progress_bar
from data import CIFAR10Dataset

class ClassiClient():
    def __init__(self, server_id, client_id, config, clients_dict, tensorboard, dev, time_str):
        self.eid = server_id
        self.cid = client_id
        self.config = config
        self.tb = tensorboard
        self.clients_dict = clients_dict
        self.dev = dev
        self.time_str = time_str

        self.best_acc = 0 
        self.epoch = 200
        self.init_lr = 0.1
        self.epoch_cnt = 0

        self.updated_model = None
        self.model = ResNet18().to(self.dev)

        self.epochs = self.config['global_round'] * self.config['EAI'] * self.config['CAI'] 

        if 'cuda' in torch.cuda.get_device_name(self.dev):
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True

        self.trainset = CIFAR10Dataset(data=self.config[self.eid][self.cid]['dataset'], mode='train')
        self.trainloader = DataLoader(self.trainset, batch_size=128, 
                shuffle=True, num_workers=2)

        self.testset = CIFAR10Dataset(data=self.config['test']['dataset'], mode='test')
        self.testloader = DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        optimizer = optim.Adam(self.model.parameters(), self.init_lr, betas = (0.9, 0.999), weight_decay = 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.dev), targets.to(self.dev)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 
                    'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        scheduler.step()
        return train_loss / len(self.trainloader), 100.*correct / total

    def test_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.dev), targets.to(self.dev)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        acc = 100.*correct/total
        # Save checkpoint.
        #if acc > self.best_acc:
            #print('Saving..')
        #    state = {
        #        'net': self.model.state_dict(),
        #        'acc': acc,
        #        'epoch': epoch,
        #    }
        #    if not os.path.isdir('checkpoint'):
        #        os.mkdir('checkpoint')
        #    torch.save(state, './checkpoint/ckpt.pth')
        #    self.best_acc = acc

        return test_loss / len(self.testloader), acc

    def train(self):
        for epoch in range(self.config['EAI']):
            if self.updated_model is not None:
                self.model.load_state_dict(self.updated_model)
                self.updated_model = None

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.test_epoch(self.epoch_cnt)

            cid = "%s.%s" % (self.eid, self.cid)
            log_info = "[%s][Epoch: %d/%d][Train.Loss: %.2f, Train.Acc: %.2f%%][Valid.Loss: %.2f, Valid.Acc: %.2f%%]" % (cid, self.epoch_cnt, self.epochs, train_loss, train_acc, val_loss, val_acc)
            print(log_info)

            self.epoch_cnt += 1

            cid = "%s.%s" % (self.eid, self.cid)
            self.tb.add_scalar(cid + '.Train.Loss', train_loss, self.epoch_cnt)
            self.tb.add_scalar(cid + '.Train.Acc', train_acc, self.epoch_cnt)
            self.tb.add_scalar(cid + '.Val.Loss', val_loss, self.epoch_cnt)
            self.tb.add_scalar(cid + '.Val.Acc', val_acc, self.epoch_cnt)

            if epoch == self.config['EAI'] - 1:
                self.clients_dict[self.cid] = self.model.state_dict()
