import os
import torch
from torch import nn
import math
import random
import shutil
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from dataset.cifar import DATASET_GETTERS, get_ood

__all__ = ['create_model', 'set_model_config',
           'set_dataset', 'set_models',
           'save_checkpoint', 'set_seed']


def create_model(args):
    if 'wideresnet' in args.arch:
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes,
                                        open=True)
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    elif args.arch == 'resnet_imagenet':
        import models.resnet_imagenet as models
        model = models.resnet18(num_classes=args.num_classes)

    return model



def set_model_config(args):

    if args.dataset == 'cifar10':
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 55
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'wideresnet_10':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == "imagenet":
        args.num_classes = 20

    args.image_size = (32, 32, 3)
    if args.dataset == 'cifar10':
        args.ood_data = ["svhn", 'cifar100', 'lsun', 'imagenet']

    elif args.dataset == 'cifar100':
        args.ood_data = ['cifar10', "svhn", 'lsun', 'imagenet']

    elif 'imagenet' in args.dataset:
        args.ood_data = ['lsun', 'dtd', 'cub', 'flowers102',
                         'caltech_256', 'stanford_dogs']
        args.image_size = (224, 224, 3)

def set_dataset(args):
    labeled_dataset, unlabeled_dataset, test_dataset, val_dataset = \
        DATASET_GETTERS[args.dataset](args)

    ood_loaders = {}
    for ood in args.ood_data:
        ood_dataset = get_ood(ood, args.dataset, image_size=args.image_size)
        ood_loaders[ood] = DataLoader(ood_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    return labeled_trainloader, unlabeled_dataset, \
           test_loader, val_loader, ood_loaders


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def set_models(args):
    model = create_model(args)
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.opt == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
    elif args.opt == 'adam':
        optimizer = optim.Adam(grouped_parameters, lr=2e-3)

    # args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    return model, optimizer, scheduler


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
