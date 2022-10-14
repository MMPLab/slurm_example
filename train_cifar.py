import os
import time
import builtins
import random
import copy
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import torch.multiprocessing as mp

import nnutils

try:
    import wandb
except ImportError:
    pass


def main(args):
    if args.seed is not None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        seed = args.seed + rank
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    args.env.distributed = args.env.world_size > 1 or (args.env.distributed and ngpus_per_node > 1)
    if args.env.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args,))
    else:
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args = copy.deepcopy(args)
    args.env.gpu = gpu
    if args.env.gpu is not None:
        print("Use GPU: {} for training".format(args.env.gpu))

    if args.env.distributed:
        if args.env.dist_url == "env://" and args.env.rank == -1:
            args.env.rank = int(os.environ["RANK"])
        args.env.rank = args.env.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.env.dist_backend, init_method=args.env.dist_url,
                                world_size=args.env.world_size, rank=args.env.rank)
        torch.distributed.barrier()

    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True

    # Prep experiment folder
    experiment_dir = f'{args.output}/{args.exp_name}'
    os.makedirs(experiment_dir, exist_ok=True)
    print(f'saving to {experiment_dir}')

    # Log all terminal outputs. Only print to terminal master process.
    is_main = not args.env.distributed or args.env.rank == 0
    print_path = os.path.join(experiment_dir, f'rank{args.env.rank:02d}.out')
    old_print = builtins.print
    def new_print(*out, **kwargs):
        if is_main: old_print(*out, **kwargs)
        open(print_path, 'a').write(' '.join([str(o) for o in out]) + '\n')
    builtins.print = new_print

    # Weights and Biases
    if args.log.use_wandb and args.env.rank == 0:
        os.makedirs(args.log.wandb_dir, exist_ok=True)
        runid = None
        if os.path.exists(f"{args.log.wandb_dir}/runid.txt"):
            runid = open(f"{args.log.wandb_dir}/runid.txt").read()
        wandb.init(project=f'cifar10-{args.arch}',
                   name=args.exp_name,
                   dir=args.log.wandb_dir,
                   entity=args.log.wandb_user,
                   resume="allow",
                   id=runid)
        open(f"{args.log.wandb_dir}/runid.txt", 'w').write(wandb.run.id)

    # Adjust parameters when training in distributed mode
    args.lr = args.lr * args.batch_size / 256   # Linear LR scaling
    if args.env.distributed and args.env.gpu is not None:
        args.batch_size = int(args.batch_size // args.env.world_size)
        args.env.workers = int((args.env.workers + ngpus_per_node - 1) / ngpus_per_node)

    global best_acc1

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=args.pretrained)
    model.fc = nn.Linear(model.fc.weight.shape[1], 10)
    model = nnutils.distribute_to_cuda(model, args.env.distributed, args.env.gpu, syncbn=False)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.env.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Learning rate to the initial LR decayed by 10 every 30 epochs
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Model checkpointing
    modules2save = {'model': model, 'optimizer': optimizer}
    ckpt_manager = nnutils.CheckpointManager(
        modules=modules2save,
        ckpt_dir=args.log.ckpt_dir,
        epochs=args.epochs,
        save_freq=args.log.save_freq)

    start_epoch = 0
    if args.resume or args.evaluate:     # resume from latest checkpoint
        start_epoch, info = ckpt_manager.resume()
        best_acc1 = info['best_acc1'] if 'best_acc1' in info else 0.

    # Data
    print('==> Preparing data.')
    if args.env.distributed and args.env.rank != 0:
        dist.barrier()
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=args.env.rank==0,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=args.env.rank==0,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    if args.env.distributed and args.env.rank == 0:
        dist.barrier()

    train_sampler = None
    val_sampler = None
    if args.env.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.env.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.env.workers, pin_memory=True, sampler=val_sampler)

    # Train and/or eval
    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(start_epoch, args.epochs):
        if args.env.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        is_best = False
        if epoch % args.log.eval_freq == 0:
            global_step = (epoch + 1) * len(train_loader)
            acc1 = validate(val_loader, model, criterion, args)
            if args.log.use_wandb and args.env.rank == 0:
                wandb.log({'Valid Acc': acc1}, step=global_step)

            # Remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        # Save checkpoint
        info = {'epoch': epoch + 1, 'best_acc1': best_acc1}
        ckpt_manager.checkpoint(epoch=epoch + 1, save_dict=info, is_best=is_best)

        scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = nnutils.AverageMeter('Time', ':6.3f')
    data_time = nnutils.AverageMeter('Data', ':6.3f')
    losses = nnutils.AverageMeter('Loss', ':.4e')
    top1 = nnutils.AverageMeter('Acc@1', ':6.2f')
    top5 = nnutils.AverageMeter('Acc@5', ':6.2f')
    progress = nnutils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.env.gpu is not None:
            images = images.cuda(args.env.gpu, non_blocking=True)
            target = target.cuda(args.env.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        losses.update(loss.item(), images.size(0))

        # measure accuracy
        acc1, acc5 = nnutils.accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log.print_freq == 0:
            progress.display(i + 1)
            if args.log.use_wandb and args.environment.rank == 0:
                global_step = i + epoch * len(train_loader)
                log_dict = {meter.name: meter.val for meter in progress.meters}
                wandb.log(log_dict, step=global_step)


def validate(val_loader, model, criterion, args):
    batch_time = nnutils.AverageMeter('Time', ':6.3f', 'none')
    losses = nnutils.AverageMeter('Loss', ':.4e', 'none')
    top1 = nnutils.AverageMeter('Acc@1', ':6.2f', 'average')
    top5 = nnutils.AverageMeter('Acc@5', ':6.2f', 'average')
    progress = nnutils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.env.gpu is not None:
                images = images.cuda(args.env.gpu, non_blocking=True)
                target = target.cuda(args.env.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))

            # measure accuracy and record loss
            acc1, acc5 = nnutils.accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log.print_freq == 0:
                progress.display(i + 1)

    if args.env.distributed:
        top1.all_reduce()
        top5.all_reduce()

    progress.display_summary()

    return top1.avg
