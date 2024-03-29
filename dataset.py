'''
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100
        MNIST
'''

import os 
import numpy as np 
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'mnist_dataloaders']


def cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10', num_workers=2):

    train_transform = transforms.Compose([
        # Data Augmentation DISABLED. Data augmentation lowers utility when training with DP

        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-10\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    # print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')
    print('Data Augmentation DISABLED!!! ')

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    sample_rate = batch_size / len(train_set)

    train_loader = DataLoader(
        train_set,
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_set),
            sample_rate=sample_rate,
        ),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, sample_rate


def cifar100_dataloaders(batch_size=128, data_dir='datasets/cifar100', num_workers=2):

    train_transform = transforms.Compose([
        # Data Augmentation DISABLED. Data augmentation lowers utility when training with DP

        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-100\t 45000 images for training \t 5000 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    # print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')
    print('Data Augmentation DISABLED!!! ')

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    sample_rate = batch_size / len(train_set)

    train_loader = DataLoader(
        train_set,
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_set),
            sample_rate=sample_rate,
        ),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, sample_rate


def mnist_dataloaders(batch_size=128, data_dir='datasets/mnist', num_workers=2):

    train_transform = transforms.Compose([
        # Data Augmentation DISABLED. Data augmentation lowers utility when training with DP

        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: MNIST\t 50000 images for training \t 10000 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data Augmentation DISABLED!!! ')

    train_set = Subset(MNIST(data_dir, train=True, transform=train_transform, download=True), list(range(50000)))
    val_set = Subset(MNIST(data_dir, train=True, transform=test_transform, download=True), list(range(50000, 60000)))
    test_set = MNIST(data_dir, train=False, transform=test_transform, download=True)

    sample_rate = batch_size / len(train_set)

    train_loader = DataLoader(
        train_set,
        num_workers=num_workers,
        pin_memory=True,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_set),
            sample_rate=sample_rate,
        ),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, sample_rate
