# encoding: utf-8

import os
import math
import torch
import shutil


def adjust_learning_rate_iter(optimizer, iters, args, ITERS_PER_EPOCH=5004):
    """Decay the learning rate based on schedule"""
    total_iters = ITERS_PER_EPOCH * args.total_epochs

    lr = args.lr
    if args.scheduler == "cos":  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    elif args.scheduler == "warmcos":
        warmup_total_iters = ITERS_PER_EPOCH * args.warmup_epochs
        if iters <= warmup_total_iters:
            lr_start = 1e-6
            lr = (lr - lr_start) * iters / float(warmup_total_iters) + lr_start
        else:
            lr *= 0.5 * (1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters)))
    elif args.scheduler == "multistep":  # stepwise lr schedule
        milestones = [int(total_iters * milestone / args.total_epochs) for milestone in [90, 120]]
        for milestone in milestones:
            lr *= 0.2 if iters >= milestone else 1.0
    elif args.scheduler == "constant":  # lr schedule
        return lr
    else:
        raise ValueError("Scheduler version {} not supported.".format(args.scheduler))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(state, is_best, save, model_name="", linear=False):
    
    if linear: model_name += '_linear'
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, model_name + "_best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)


def load_checkpoint(args, file_name, model, optimizer, linear=False, tea=False):
    start_epoch = 1
    
    save_file = os.path.join(file_name, 'last_epoch_ckpt.pth.tar')
    if not linear and os.path.isfile(save_file):
        state = torch.load(save_file, map_location='cpu')
        start_epoch = state['start_epoch'] + 1
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        del state        
        if args.rank == 0: print("=> loaded successfully, training starts from (epoch {})".format(start_epoch))
        
    elif linear:
        save_file_linear = os.path.join(file_name, 'last_epoch_linear_ckpt.pth.tar')
        if os.path.isfile(save_file_linear):
            state = torch.load(save_file_linear, map_location='cpu')
            start_epoch = state['start_epoch'] + 1
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            del state
            if args.rank == 0: print("=> loaded successfully, training starts from (epoch {})".format(start_epoch))

        elif os.path.isfile(save_file):
            state = torch.load(save_file, map_location='cpu')            
            new_state = {}
            for k, v in state['model'].items():
                if (not tea) and ('student' in k) and ('student_ema' not in k):
                    new_state[k.replace("student", "encoder")] = v
                elif tea and ('teacher' in k) and ('teacher_ema' not in k):
                    new_state[k.replace("teacher", "encoder")] = v
        
            model.load_state_dict(new_state, strict=False)
            del state
            if args.rank == 0: print("=> loaded successfully, training starts from (epoch {})".format(start_epoch))

        else:
            
            if args.rank == 0: print("=> no checkpoint found from ", save_file)
    else:
        if args.rank == 0: print("=> no checkpoint found from ", save_file)
    
    return start_epoch, model, optimizer


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_devices(gpu_ids):
    if "-" in gpu_ids:
        gpus = gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        parsed_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))
        return parsed_ids
    else:
        return gpu_ids


class AvgMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()
        self.val = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
