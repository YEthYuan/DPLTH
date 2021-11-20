'''
    setup model and datasets
'''




import copy 
import torch
import torch.nn as nn
import numpy as np 
from advertorch.utils import NormalizeByChannelMeanStd

from models import *
from dataset import *


__all__ = ['setup_model_dataset']


def setup_model_dataset(args):

    if args.dataset == 'cifar10':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader, sample_rate = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)

    elif args.dataset == 'cifar100':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, val_loader, test_loader, sample_rate = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)

    elif args.dataset == 'mnist':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.1307, ], std=[0.3081, ])
        train_set_loader, val_loader, test_loader, sample_rate = mnist_dataloaders(batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
    
    else:
        raise ValueError('Dataset not supprot yet !')

    # first load the pretrained model supervised by Imagenet (1,000 classes)
    downstream_classes = classes
    if args.pre_train:
        classes = 1000

    if args.imagenet_arch or args.pre_train:
        model = model_dict[args.arch](num_classes=classes, imagenet=True, pretrained=args.pre_train)
    else:
        model = model_dict[args.arch](num_classes=classes, pretrained=args.pre_train)

    # then reset the final fc layer
    if args.pre_train:
        if args.arch == 'resnet18':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, downstream_classes)

    model.normalize = normalization
    if args.pre_train:
        print(f'*********** Weights loaded from the pretrained {args.arch} ***********')

    print(model)

    return model, train_set_loader, val_loader, test_loader, sample_rate
