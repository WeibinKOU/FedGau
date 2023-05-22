import argparse
import numpy as np
import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import sys
import imgaug.augmenters as iaa
from config import *
from MultiFL.fed_server import CloudServer
from MultiFL.fed_scheduler import Scheduler

from model.bisenetv2 import BiSeNetV2
from model.cracknet import CrackNet
from model.segnet import SegNet
from model.deeplabv3 import DeepLabv3

def print_device_info():
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = False
        print(torch.version.cuda)
        print(torch.backends.cudnn.version())
        print(torch.cuda.get_device_name(0))

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.003, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--gpu", type=int, default=0, help="the index of GPU used to train")
    parser.add_argument("--disable_pretrain", action='store_true', help="whether to enable pretaining stage to initialize all the models of edges and vehicles")
    parser.add_argument("--pretrain_epochs", type=int, default=100, help="number of epochs of pretraining")
    parser.add_argument("--epochs_after_pretrain", type=int, default=500, help="number of epochs of each vehicle after pretrain")
    parser.add_argument("--edge_fed_interval", type=int, default=1, help="each edge_fed_interval vehicle training to do a Edge Server federated learning")
    parser.add_argument("--cloud_fed_interval", type=int, default=1, help="each cloud_fed_interval Edge Server federated learning to do a Cloud Server federated learning")
    parser.add_argument("--total_size", type=int, default=30, help="total size of communication resource")
    parser.add_argument("--model", type=str, default='CrackNet', help="To specify which model to be trained, options: [CrackNet, BiSeNetV2, SegNet, DeepLabv3]")
    args = parser.parse_args()
    return args

def save_cmd(log_dir):
    txt_file = open(log_dir +'/cmd.txt', 'w')
    cmd=" ".join("\"" + arg + "\"" if " " in arg else arg for arg in sys.argv)
    cmd = 'python ' + cmd
    txt_file.write(cmd)
    txt_file.close()

def main():
    print_device_info()
    tb = SummaryWriter()
    save_cmd(tb.logdir)
    args = build_parser()

    training_config = {}
    training_config['epochs'] = args.epochs_after_pretrain
    training_config['batch_size'] = args.batch_size
    training_config['lr'] = args.lr
    training_config['betas'] = (args.b1, args.b2)
    training_config['weight_decay'] = 1e-4

    aug_seq = iaa.Sequential([
        iaa.Fliplr(.5),
        iaa.Affine(
            translate_percent={'x': (-.05, .05), 'y': (-.05, .05)},
            rotate=(-25, 25)
        ),

        #iaa.GammaContrast((.4, 2.5)),
        #iaa.GaussianBlur((0, 3.0)),

        iaa.Resize({'height': HEIGHT, 'width': WIDTH}),
    ])

    scheduler = Scheduler(-1, 14.8, 400)

    scheduler.set_edge_fed_interval(args.edge_fed_interval)
    scheduler.set_cloud_fed_interval(args.cloud_fed_interval)
    scheduler.set_pretrain_epochs(args.pretrain_epochs)
    scheduler.set_epochs_after_pretrain(args.epochs_after_pretrain)

    platform = sys.platform
    if 'win' in platform:
        log_dir = tb.logdir.split('\\')[-1]
    elif 'linux' in platform:
        log_dir = tb.logdir.split('/')[-1]

    model_sel = {}
    if args.model == 'CrackNet':
        model_sel['model'] = CrackNet
    elif args.model == 'BiSeNetV2':
        model_sel['model'] = BiSeNetV2
    elif args.model == 'SegNet':
        model_sel['model'] = SegNet
    elif args.model == 'DeepLabv3':
        model_sel['model'] = DeepLabv3

    dev = torch.device('cuda:%d' % (args.gpu) if torch.cuda.is_available() else 'cpu')

    cloud = CloudServer(aug_seq, training_config, tb, scheduler, log_dir, model_sel, dev)

    try:
        if not args.disable_pretrain:
            cloud.edges[0].clients[0].pretrain()
            cloud.avgModel = cloud.edges[0].clients[0].model.state_dict()
        cloud.SinkModelToEdges()
        for edge in cloud.edges:
            edge.SinkModelToClients()
        cloud.run()
    finally:
        tb.close()

if __name__ == '__main__':
    main()
