import os
from enum import Enum
import torch
import torch.distributed as dist
import shutil

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type='average'):
        self.name = name
        self.fmt = fmt
        if summary_type == 'average':
            self.summary_type = Summary.AVERAGE
        elif summary_type == 'none':
            self.summary_type = Summary.NONE
        elif summary_type == 'sum':
            self.summary_type = Summary.SUM
        elif summary_type == 'count':
            self.summary_type = Summary.COUNT
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

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        if dist.is_available() and dist.is_initialized():
            total = total.cuda()
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def distribute_to_cuda(model, distributed=True, gpu=0, syncbn=False):
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif distributed:
        if syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpu is not None:
            torch.cuda.set_device(gpu)
            model.cuda(gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise NotImplementedError
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class CheckpointManager:
    def __init__(self,
                 modules,
                 ckpt_dir,
                 epochs,
                 save_freq=None):
        self.modules = modules
        self.ckpt_dir = ckpt_dir
        self.epochs = epochs
        self.save_freq = save_freq
        self.retain_num_ckpt = 0

        self.distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0
        os.makedirs(os.path.join(self.ckpt_dir), exist_ok=True)

    def resume(self):
        ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_latest.pth')
        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')
            for k in self.modules:
                self.modules[k].load_state_dict(checkpoint[k])
            start_epoch = checkpoint['epoch']
            info = {k: checkpoint[k] for k in checkpoint if k not in self.modules}
            print(f"=> loaded checkpoint '{ckpt_fname}' (epoch {checkpoint['epoch']})")
        else:
            start_epoch = 0
            info = {}
            print(f"=> no checkpoint found at '{ckpt_fname}'")
        return start_epoch, info

    def create_state_dict(self, save_dict):
        state = {k: self.modules[k].state_dict() for k in self.modules}
        if save_dict is not None:
            state.update(save_dict)
        return state

    def checkpoint(self, epoch, save_dict=None, is_best=False):
        state = self.create_state_dict(save_dict)
        if self.rank != 0:  # Only save on master process
            return

        latest_fname = os.path.join(self.ckpt_dir, f'checkpoint_latest.pth')
        save_checkpoint(state, is_best=is_best, filename=latest_fname)
        print(f"=> saved checkpoint '{latest_fname}' (epoch {epoch})")

        if (epoch % self.save_freq == 0) or epoch == self.epochs:
            ckpt_fname = os.path.join(self.ckpt_dir, f'checkpoint_{epoch:04d}.pth')
            shutil.copyfile(latest_fname, ckpt_fname)
            print(f"=> saved checkpoint '{ckpt_fname}' (epoch {epoch})")
