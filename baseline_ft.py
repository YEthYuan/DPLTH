'''
    main process for the DP-SGD training baseline I (Supervised pre-trained network + DP fine-tune at fully train mode)
    more about ft scenario: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''
import os
import pdb
import time
import pickle
import random
import shutil
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
from torch.nn import modules
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd
from opacus import PrivacyEngine
from opacus.layers import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils import stats
from opacus.utils.module_modification import convert_batchnorm_modules
from opacus.dp_model_inspector import DPModelInspector

from utils import *
from pruner import *

parser = argparse.ArgumentParser(description='Differentially Private Transfer Learning Experiments')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset, cifar10 or mnist')
parser.add_argument('--input_size', type=int, default=32, help='mnist=28, cifar10=32')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
# parser.add_argument('--pre_train', action="store_true", help='imagenet supervised pre-train model')
parser.add_argument('--pre_train', type=bool, default=True, help='imagenet supervised pre-train model')
parser.add_argument('--imagenet_arch', action="store_true", help="architecture for imagenet size samples")

##################################### General setting ############################################
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='./record_without_name_please_modify_its_name', type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--virtual_batch_size', type=int, default=512, help='vbs/bs == accumulation_steps')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### DP-SGD setting #################################################
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='clip the gradient by scaling it down to this value')
parser.add_argument('--eps', type=float, default=50.0, help='epsilon is a privacy budget of a DP-SGD algorithm')
parser.add_argument('--delta', type=float, default=1e-5, help='delta is always set to 1e-5 by default')

best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    print(args)

    assert args.virtual_batch_size % args.batch_size == 0 # VIRTUAL_BATCH_SIZE should be divisible by BATCH_SIZE
    accumulation_steps = int(args.virtual_batch_size / args.batch_size)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset 
    model, train_loader, val_loader, test_loader, sample_rate = setup_model_dataset(args)
    inspector = DPModelInspector()
    # if the model is incompatible with the privacy engine
    if True:
        print(f'BatchNorm layers in {args.arch} is incompatible with the privacy engine, automatically replaced with GroupNorm')
        model = convert_batchnorm_modules(model)

    # check again if the model is legal
    inspector.validate(model)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    # if args.prune_type == 'lt':
    #     print('lottery tickets setting (rewind to the same random init)')
    #     initalization = deepcopy(model.state_dict())
    # elif args.prune_type == 'pt':
    #     print('lottery tickets from best dense weight (pretrained)')
    #     initalization = None
    # elif args.prune_type == 'rewind_lt':
    #     print('lottery tickets with early weight rewinding')
    #     initalization = None
    # else:
    #     raise ValueError('unknown prune_type')

    #
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=sample_rate * accumulation_steps,
        epochs=args.epochs,
        target_epsilon=args.eps,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
    )
    privacy_engine.attach(optimizer)
    print(f"Using sigma={privacy_engine.noise_multiplier} and C={args.max_grad_norm}")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    # Sorry that we do not support resume currently
    # if args.resume:
    #     print('resume from checkpoint {}'.format(args.checkpoint))
    #     checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
    #     best_sa = checkpoint['best_sa']
    #     start_epoch = checkpoint['epoch']
    #     all_result = checkpoint['result']
    #     start_state = checkpoint['state']
    #
    #     if start_state>0:
    #         current_mask = extract_mask(checkpoint['state_dict'])
    #         prune_model_custom(model, current_mask)
    #         check_sparsity(model)
    #         optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                                     momentum=args.momentum,
    #                                     weight_decay=args.weight_decay)
    #         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    #
    #     model.load_state_dict(checkpoint['state_dict'])
    #     # adding an extra forward process to enable the masks
    #     x_rand = torch.rand(1,3,args.input_size, args.input_size).cuda()
    #     with torch.no_grad:
    #         model(x_rand)
    #
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     initalization = checkpoint['init_weight']
    #     print('loading state:', start_state)
    #     print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)
    #
    # else:
    #     all_result = {}
    #     all_result['train_ta'] = []
    #     all_result['test_ta'] = []
    #     all_result['val_ta'] = []
    #
    #     start_epoch = 0
    #     start_state = 0

    all_result = {}
    all_result['train_ta'] = []
    all_result['test_ta'] = []
    all_result['val_ta'] = []
    all_result['epsilon'] = []
    all_result['alpha'] = []

    start_epoch = 0
    # start_state = 0

    print('######################################## Start Fine-Tuning the Pretrained Model ########################################')
    
    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc, privacy_budget = train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, privacy_engine=privacy_engine, epoch=epoch)

        # if state == 0:
        #     if (epoch+1) == args.rewind_epoch:
        #         torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1)))
        #         if args.prune_type == 'rewind_lt':
        #             initalization = deepcopy(model.state_dict())

        # evaluate on validation set
        print('evaluate on validation set')
        tacc = validate(val_loader, model, criterion, privacy_engine)
        # evaluate on test set
        print('evaluate on test set')
        test_tacc = validate(test_loader, model, criterion, privacy_engine)

        scheduler.step()

        all_result['train_ta'].append(acc)
        all_result['val_ta'].append(tacc)
        all_result['test_ta'].append(test_tacc)
        all_result['epsilon'].append(privacy_budget['epsilon'])
        all_result['alpha'].append(privacy_budget['best_alpha'])

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint({
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_sa': best_sa,
            'optimizer': optimizer.state_dict(),
            'privacy_engine': privacy_engine.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_SA_best=is_best_sa, save_path=args.save_dir)

        # plot training curve
        plt.plot(all_result['train_ta'], label='train_acc')
        plt.plot(all_result['val_ta'], label='val_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train_acc.png'))
        plt.close()

        # plot privacy curve
        plt.plot(all_result['epsilon'], label='ε')
        plt.plot(all_result['alpha'], label='α')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train_privacy.png'))
        plt.close()

        #report result
        val_pick_best_epoch = np.argmax(np.array(all_result['val_ta']))
        print('* best SA = {}, Epoch = {}'.format(all_result['test_ta'][val_pick_best_epoch], val_pick_best_epoch+1))


def train(train_loader, model, criterion, optimizer, privacy_engine, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    epsilon, best_alpha = privacy_engine.get_privacy_spent() # final epsilon

    print(
        f"Train Accuracy: {top1.avg:.3f}\t"
        f"Train Loss: {losses.avg:.6f}\t"
        f"(ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
    )

    privacy_indicators = {
        'epsilon': epsilon,
        'delta': privacy_engine.target_delta,
        'best_alpha': best_alpha
    }

    return top1.avg, privacy_indicators

def validate(val_loader, model, criterion, privacy_engine):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    epsilon, best_alpha = privacy_engine.get_privacy_spent()
    print(
        f"Validation Accuracy: {top1.avg:.3f}\t"
        f"Validation Loss: {losses.avg:.6f}\t"
        f"(ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
    )

    return top1.avg


def save_checkpoint(state, is_SA_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))


def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def setup_seed(seed): 
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


if __name__ == '__main__':
    main()


