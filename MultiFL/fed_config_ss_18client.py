#you can manually add more edge server and agents repectively by appending items

from model.bisenetv2 import BiSeNetV2
from model.cracknet import CrackNet
from model.segnet import SegNet
from model.deeplabv3 import DeepLabv3
from model.fasterrcnn.frcnn import FasterRCNN
from model.resnet import ResNet18
import torch.nn.functional as F

from utils.func import *

WIDTH=320
HEIGHT=320

task = 'semSeg' #alternatives: semSeg, objDect, classification
semSeg = {
        'Edge0' : #Northern Germany
        {
            'agg_coef': 0.255,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/hamburg/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.416,
            },

            'Agent1' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/hanover/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.326,
            },

            'Agent2' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/bremen/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.258,
            },
        },

        'Edge1': # Rhine-Ruhr Region
        {
            'agg_coef': 0.097,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/bochum/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.332,
            },

            'Agent1' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/krefeld/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.343,
            },

            'Agent2' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/monchengladbach/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.325,
            },
        },

        'Edge2': #Western Germany - Cental
        {
            'agg_coef': 0.126,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/cologne/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.411,
            },

            'Agent1' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/dusseldorf/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.589,
            }
        },

        'Edge3': # Sourthern Germany - East
        {
            'agg_coef': 0.080,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/ulm/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.603,
            },

            'Agent1' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/tubingen/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.397,
            }
        },

        'Edge4': # Central Germany - East
        {
            'agg_coef': 0.124,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/erfurt/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.295,
            },

            'Agent1' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/jena/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.322,
            },

            'Agent2' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/weimar/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 0.384,
            },
        },

        'Edge5': # Zurich
        {
            'agg_coef': 0.041,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/zurich/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 1.0,
            },
        },

        'Edge6': # Sourthern Germany - West
        {
            'agg_coef': 0.066,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/stuttgart/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 1.0,
            },
        },

        'Edge7': # Western Germany - North
        {
            'agg_coef': 0.058,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/aachen/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 1.0,
            },
        },

        'Edge8': # Central Germany - West
        {
            'agg_coef': 0.029,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/darmstadt/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 1.0,
            },
        },

        'Edge9': # France
        {
            'agg_coef': 0.123,
            'Agent0' :
            {
                'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/strasbourg/',
                'lr': 0.003,
                'batch_size': 4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4,
                'agg_coef': 1.0,
            },
        },

        'test':
        {
            'dataset' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/cityspace/test/',
            'batch_size': 4,
        },

        'model': DeepLabv3,
        'global_round': 50,
        'EAI': 1,
        'CAI': 1,
        'GPU_ID': 0,
        'logdir': '/home/wbkou/AAAI/HFL-DynaCoeffi/logs'
}


objDect = {
        'Edge0' :
        {
            'agg_coef': 0.175,
            'Agent0' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_1.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_1.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                'agg_coef': 0.138,
            },

            'Agent1' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_2.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_2.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                'agg_coef': 0.862,
            }
        },

        'Edge1':
        {
            'agg_coef': 0.394,
            'Agent0' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_3.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_3.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                'agg_coef': 0.075,
            },

            'Agent1' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_4.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_4.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                'agg_coef': 0.446,
            },

            'Agent2' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_5.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_5.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                'agg_coef': 0.479,
            }
        },

        'Edge2':
        {
            'agg_coef': 0.431,
            'Agent0' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_6.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_6.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                'agg_coef': 0.529,
            },

            'Agent1' :
            {
                #'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/VOCdevkit/repack/2007_train_7.txt',
                'train_anno_path' : '/home/wbkou/AAAI/HFL-DynaCoeffi/datasets/kitti/repack/train_7.txt',
                'batch_size': 8,
                'lr': 0.003,
                'betas': (0.9, 0.999),
                'weight_decay': 0,
                'agg_coef': 0.471,
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
