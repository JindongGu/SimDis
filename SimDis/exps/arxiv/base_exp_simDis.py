# encoding: utf-8

import os
import itertools

import torch
import torch.nn as nn
import torch.distributed as dist

from SimDis.models.sim_dis_train_model import SimDis_Model
from SimDis.exps.arxiv import base_exp
from SimDis.layers.optimizer import LARS_SGD

class Exp(base_exp.BaseExp):
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args

        # ------------------------------------- model config ------------------------------ #
        self.param_momentum = args.ema_moment

        # ------------------------------------ data loader config ------------------------- #
        self.data_num_workers = 10

        # ------------------------------------  training config --------------------------- #
        self.warmup_epochs = 10
        self.max_epoch = args.epochs
        self.warmup_lr = 1e-6
        self.basic_lr_per_img = args.basic_lr / 256.0
        self.lr = self.basic_lr_per_img * args.word_size * args.nr_gpu * args.batchsize

        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.print_interval = 200
        self.n_views = args.n_views

        self.exp_name = '{}_stu_{}_tea_{}_ema_{}_lr_{}_syncBN_{}_opt_{}_epoch_{}_BS_{}_GPUs_{}'.format(
            args.method, args.model_s, args.model_t, args.ema_moment, self.lr,
            args.syncBN, args.optimizer, args.epochs, args.batchsize, args.word_size * args.nr_gpu
        )    

    def get_model(self):
        if "model" not in self.__dict__:
            self.model = SimDis_Model(self.args, self.param_momentum, len(self.data_loader["train"]) * self.max_epoch)
        return self.model
    

    def get_data_loader(self, batch_size, is_distributed, if_transformer=False):
        if "data_loader" not in self.__dict__:
            if if_transformer:
                pass
            else:            
                from SimDis.data.transforms import byol_transform
                from SimDis.data.dataset import SSL_Dataset

                transform = byol_transform()        
                train_set = SSL_Dataset(transform)

            sampler = None

            if is_distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(train_set)

            dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": False}
            dataloader_kwargs["sampler"] = sampler
            dataloader_kwargs["batch_size"] = batch_size
            dataloader_kwargs["shuffle"] = False
            dataloader_kwargs["drop_last"] = True
            train_loader = torch.utils.data.DataLoader(train_set, **dataloader_kwargs)
            self.data_loader = {"train": train_loader, "eval": None}
            
        return self.data_loader
    

    def get_optimizer(self, model, batch_size):
        # Noticing hear we only optimize student_encoder
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.lr

            paras = []
            if self.args.model_s is not None: paras += list(model.student.parameters())
            if (self.args.model_t is not None) and (not self.args.offline) : paras += list(model.teacher.parameters())

            if self.args.optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(paras, lr=lr, weight_decay=self.weight_decay, momentum=self.momentum)
                if self.args.rank == 0: print(self.args.optimizer, 'Optimizer is used!')
                
            elif self.args.optimizer == 'LARS':
                params_lars = []
                params_exclude = []
                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                        params_exclude.append(m.weight)
                        params_exclude.append(m.bias)
                    elif isinstance(m, nn.Linear):
                        params_lars.append(m.weight)
                        params_exclude.append(m.bias)
                    elif isinstance(m, nn.Conv2d):
                        params_lars.extend(list(m.parameters()))

                assert len(params_lars) + len(params_exclude) == len(list(self.model.parameters()))
                
                self.optimizer = LARS_SGD(
                    [{"params": params_lars, "lars_exclude": False}, {"params": params_exclude, "lars_exclude": True}],
                    lr=lr,
                    weight_decay=self.weight_decay,
                    momentum=self.momentum,
                )
                if self.args.rank == 0: print(self.args.optimizer, 'Optimizer is used!')
                
        return self.optimizer


