#§ encoding: utf-8

import torch
from torch import nn
import torch.distributed as dist

from SimDis.models import resnet_mbn as resnet
from SimDis.exps.arxiv.base_exp import BaseExp
from SimDis.utils import accuracy, AvgMeter
from SimDis.utils.torch_dist import reduce_tensor_sum

class ResNetWithLinear(nn.Module):
    def __init__(self, args):
        super(ResNetWithLinear, self).__init__()

        self.args = args

        if args.syncBN: bn = "torchsync"
        else: bn = "vanilla"
        
        if args.rank == 0: print(bn, 'Batch Normalization is used! ', args.model_s, 'is to be trained!')

        self.encoder = resnet.load_resnet(args.model_s)(
                low_dim=256, width=1, hidden_dim=4096, MLP="byol", CLS=False, bn=bn, predictors=args.model_s.split('.')[1:])

        for p in self.encoder.parameters():
            p.requires_grad = False

        expansion = 4
        if ('resnet18' in args.model_s) or ('resnet34' in args.model_s): expansion = 1
        feat_inp_dim = 512 * expansion
            
        self.classifier = nn.Sequential(nn.Linear(feat_inp_dim, 1000), nn.BatchNorm1d(1000))

        self.criterion = nn.CrossEntropyLoss()
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def train(self, mode: bool = True):
        self.training = mode
        self.encoder.eval()
        self.classifier.train(mode)

    def forward(self, x, target=None):
        with torch.no_grad():
            feat = self.encoder(x, feat=True).detach()

        logits = self.classifier(feat)
        if self.training:
            loss = self.criterion(logits, target)
            return loss
        else:
            return logits


class Exp(BaseExp):
    def __init__(self, args):
        super(Exp, self).__init__()

        # ----------------------------- moco setting ------------------------------- #
        self.basic_lr_per_img = 30.0 / 256.0
        self.max_epochs = 100
        self.scheduler = "cos"
        self.milestones = [60, 80]
        self.save_folder_prefix = ""
        self.warmup_epochs = 0

        self.args = args
        self.exp_name = args.method

    def get_model(self):
        if "model" not in self.__dict__:
            self.model = ResNetWithLinear(self.args)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, if_transformer=False):
        if "data_loader" not in self.__dict__:
            
            from SimDis.data.dataset import ImageNet
            from SimDis.data.transforms import typical_imagenet_transform

            train_set = ImageNet(True, typical_imagenet_transform(True))
            eval_set = ImageNet(False, typical_imagenet_transform(False))

            train_dataloader_kwargs = {
                "num_workers": 6,
                "pin_memory": False,
                "batch_size": self.args.batchsize,
                "shuffle": False,
                "drop_last": True,
                "sampler": torch.utils.data.distributed.DistributedSampler(train_set) if is_distributed else None,
            }
            
            train_loader = torch.utils.data.DataLoader(train_set, **train_dataloader_kwargs)
            
            eval_loader = torch.utils.data.DataLoader(
                eval_set,
                batch_size=100,
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                drop_last=False,
                sampler=torch.utils.data.distributed.DistributedSampler(eval_set) if is_distributed else None,
            )
            
            self.data_loader = {"train": train_loader, "eval": eval_loader}
            
        return self.data_loader

    def get_optimizer(self, model, batch_size):
        if "optimizer" not in self.__dict__:
            args = self.args
            
            lr = self.basic_lr_per_img * args.word_size * args.nr_gpu * args.batchsize
            
            self.optimizer = torch.optim.SGD(
                self.model.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=False
            )
        return self.optimizer

    def run_eval(self, model, eval_loader):

        top1 = AvgMeter()
        top5 = AvgMeter()

        with torch.no_grad():
            for i, (inp, target) in enumerate(eval_loader):
                inp = inp.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                logits = model(inp)
                acc1, acc5 = accuracy(logits, target, (1, 5))
                acc1, acc5 = (
                    reduce_tensor_sum(acc1) / dist.get_world_size(),
                    reduce_tensor_sum(acc5) / dist.get_world_size(),
                )
                top1.update(acc1.item(), inp.size(0))
                top5.update(acc5.item(), inp.size(0))

        if self.args.rank == 0:
            print("Accu, top1: {:.4f}, top5: {:.4f}".format(top1.avg, top5.avg))


            
