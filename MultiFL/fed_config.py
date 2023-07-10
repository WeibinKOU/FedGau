#you can manually add more edge server and agents repectively by appending items

from model.bisenetv2 import BiSeNetV2
from model.cracknet import CrackNet
from model.segnet import SegNet
from model.deeplabv3 import DeepLabv3
from model.fasterrcnn.frcnn import FasterRCNN
from model.resnet import ResNet18
import torch.nn.functional as F

from utils.func import *

#WIDTH=1280 #cityscapes
#HEIGHT=640 #cityscapes

WIDTH=960 #Mapillary
HEIGHT=720 #Mapillary

#WIDTH=960 #CamVid
#HEIGHT=720 #CamVid

task = 'semSeg' #alternatives: semSeg, objDect, classification
semSeg = {
        'Edge0' : #Northern Germany
        {
            #'agg_coef': 0.363, #cityscapes fedavg
            #'agg_coef': 0.41, #cityscapes fedstats
            'agg_coef': 0.41, #Mapillary fedavg
            #'agg_coef': 0.45, #Mapillary fedstats
            #'agg_coef': 0.47, #CamVid fedavg
            #'agg_coef': 0.59, #CamVid fedstats
            'Agent0' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/client4/',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/Mapillary/client4/',
                'lr': 0.0001,
                'batch_size': 8,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                #'agg_coef': 0.535, #cityscapes fedavg
                #'agg_coef': 0.41, #cityscapes fedstats
                'agg_coef': 0.58, #Mapillary fedavg
                #'agg_coef': 0.79, #Mapillary fedstats
                #'agg_coef': 0.37, #CamVid fedavg
                #'agg_coef': 0.28, #CamVid fedstats
            },

            'Agent1' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/client6/',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/Mapillary/client6/',
                'lr': 0.0001,
                'batch_size': 8,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                #'agg_coef': 0.265, #cityscapes fedavg
                #'agg_coef': 0.44, #cityscapes fedstats
                'agg_coef': 0.11, #Mapillary fedavg
                #'agg_coef': 0.05, #Mapillary fedstats
                #'agg_coef': 0.11, #CamVid fedavg
                #'agg_coef': 0.07, #CamVid fedstats
            },

            'Agent2' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/client7/',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/Mapillary/client7/',
                'lr': 0.0001,
                'batch_size': 8,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                #'agg_coef': 0.201, #cityscapes fedavg
                #'agg_coef': 0.15, #cityscapes fedstats
                'agg_coef': 0.31, #Mapillary fedavg
                #'agg_coef': 0.16, #Mapillary festats
                #'agg_coef': 0.51, #CamVid fedavg
                #'agg_coef': 0.66, #CamVid fedstats
            },
        },

        'Edge1': # Rhine-Ruhr Region
        {
            #'agg_coef': 0.352, #cityscapes fedavg
            #'agg_coef': 0.11, #cityscapes festats
            'agg_coef': 0.24, #Mapillary fedavg
            #'agg_coef': 0.20, #Mapillary fedstats
            #'agg_coef': 0.25, #CamVid fedavg
            #'agg_coef': 0.19, #CamVid fedstats
            'Agent0' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/client1/',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/Mapillary/client1/',
                'lr': 0.0001,
                'batch_size': 8,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                #'agg_coef': 0.560, #cityscapes fedavg
                #'agg_coef': 0.61, #cityscapes fedstats
                'agg_coef': 0.77, #Mapillary fesavg
                #'agg_coef': 0.94, #Mapillary fedstats
                #'agg_coef': 0.79, #CamVid fedavg
                #'agg_coef': 0.97, #CamVid fedstats
            },

            'Agent1' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/client2/',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/Mapillary/client2/',
                'lr': 0.0001,
                'batch_size': 8,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                #'agg_coef': 0.440, #cityscapes fedavg
                #'agg_coef': 0.39, #cityscapes fedstats
                'agg_coef': 0.23, #Mapillary fedavg
                #'agg_coef': 0.06, #Mapillary fedstats
                #'agg_coef': 0.21, #CamVid fedavg
                #'agg_coef': 0.03, #CamVid fedstats
            },
        },

        'Edge2': #Western Germany - Cental
        {
            #'agg_coef': 0.285, #cityscapes fedavg
            #'agg_coef': 0.48, #cityscapes fedstats
            'agg_coef': 0.35, #Mapillary fedavg
            #'agg_coef': 0.35, #Mapillary fedstats
            #'agg_coef': 0.28, #CamVid fedavg
            #'agg_coef': 0.22, #CamVid fedstats
            'Agent0' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/client3/',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/Mapillary/client3/',
                'lr': 0.0001,
                'batch_size': 8,
               'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                #'agg_coef': 0.652, #cityscapes fedavg
                #'agg_coef': 0.90, #cityscapes fedstats
                'agg_coef': 0.29, #Mapillary fedavg
                #'agg_coef': 0.12, #Mapillary fedstats
                #'agg_coef': 0.33, #CamVid fedavg
                #'agg_coef': 0.17, #CamVid fedstats
            },

            'Agent1' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/client5/',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/Mapillary/client5/',
                'lr': 0.0001,
                'batch_size': 8,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                #'agg_coef': 0.348, #cityscapes fedavg
                #'agg_coef': 0.10, #cityscapes fedstats
                'agg_coef': 0.71, #Mapillary fedavg
                #'agg_coef': 0.88, #Mapillary fedstats
                #'agg_coef': 0.67, #CamVid fedavg
                #'agg_coef': 0.83, #CamVid fedstats
            }
        },

        'test':
        {
            #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/test/',
            'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/Mapillary/test/',
            'batch_size': 8,
        },

        'model': DeepLabv3,
        'dataset': 'Mapillary',  #Alternatives: Mapillary or cityscapes
        'global_round': 50,
        'EAI': 3,
        'CAI': 2,
        'GPU_ID': 0,
        'enable_optim': False,
        'FedAlgo': 'FedAvg', #Alternatives: FedAvg, FedProx, FedDyn and FedStats
        'logdir': '/home/wbkou/AAAI/HFL-DynaCoeffi/logs'
}


objDect = {
        'Edge0' :
        {
            #'agg_coef': 0.175, #size ratio
            'agg_coef': 0.135, #bd
            'Agent0' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_1.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_1.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                #'agg_coef': 0.138, #size ratio
                'agg_coef': 0.008, #bd
            },

            'Agent1' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_2.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_2.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                #'agg_coef': 0.862,#size ratio
                'agg_coef': 0.992,#bd
            }
        },

        'Edge1':
        {
            #'agg_coef': 0.394,#size ratio
            'agg_coef': 0.393,#bd
            'Agent0' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_3.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_3.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                #'agg_coef': 0.075,#size ratio
                'agg_coef': 0.066,#bd
            },

            'Agent1' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_4.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_4.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                #'agg_coef': 0.446,#size ratio
                'agg_coef': 0.421,#bd
            },

            'Agent2' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_5.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_5.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                #'agg_coef': 0.479,#size ratio
                'agg_coef': 0.513,#bd
            }
        },

        'Edge2':
        {
            #'agg_coef': 0.431,#size ratio
            'agg_coef': 0.472,#size ratio
            'Agent0' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_6.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_6.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                #'agg_coef': 0.529,#size ratio
                'agg_coef': 0.577,#bd
            },

            'Agent1' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_7.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_7.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                #'agg_coef': 0.471,#size ratio
                'agg_coef': 0.423,#bd
            }
        },

        'test':
        {
            #'val_anno_path': '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/2007_val.txt',
            'val_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/val.txt',
            'batch_size': 8,
        },

        'model': FasterRCNN,
        'global_round': 50,
        'EAI': 3,
        'CAI': 2,
        'GPU_ID': 0,

        #'classes_path': '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/voc_classes.txt',
        'classes_path': '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/cls_classes.txt',
        'logdir': '/home/wbkou/AAAI/HFL-DynaCoeffi/logs',
        'lr_decay_type': 'cos',
        'save_period': 10,
        'eval_period': 10000,
        'input_shape': [200, 600],
        'anchors_size': [4, 16, 32],
}


classi = {
        'Edge0' :
        {
            'Agent0' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR10/repack/data_batch_1',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR101/new_batch_1',
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR102/new_batch_1',
                'batch_size': 32,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
            },

            'Agent1' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR10/repack/data_batch_2',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR101/new_batch_2',
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR102/new_batch_2',
                'batch_size': 32,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
            }
        },

        'Edge1':
        {
            'Agent0' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR10/repack/data_batch_3',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR101/new_batch_3',
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR102/new_batch_3',
                'batch_size': 32,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
            },

            'Agent1' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR10/repack/data_batch_4',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR101/new_batch_4',
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR102/new_batch_4',
                'batch_size': 32,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
            },

            'Agent2' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR10/repack/data_batch_5',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR101/new_batch_5',
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR102/new_batch_5',
                'batch_size': 32,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
            }
        },

        'Edge2':
        {
            'Agent0' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR10/repack/data_batch_6',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR101/new_batch_6',
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR102/new_batch_6',
                'batch_size': 32,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
            },

            'Agent1' :
            {
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR10/repack/data_batch_7',
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR101/new_batch_7',
                #'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR102/new_batch_7',
                'batch_size': 32,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
            }
        },

        'test':
        {
            'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/CIFAR10/test_batch',
            'batch_size': 32,
        },

        'model': ResNet18,
        'global_round': 50,
        'EAI': 3,
        'CAI': 2,
        'GPU_ID': 0,
        'logdir': '/home/wbkou/AAAI/HFL-DynaCoeffi/logs',
        #'Nonlinear': F.relu
        'Nonlinear': F.softsign
        #'Nonlinear': F.leaky_relu
}
