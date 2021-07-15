# encoding: utf-8

import math

import torch
import torch.nn.functional as F
from torch.nn import Module

from . import resnet_mbn as resnet

# loss for ablation study
def pred_feat_loss(q1, k1, q2, k2):
    return (4 - 2 * ((q1 * k1.detach()).sum(dim=-1, keepdim=True) + (q2 * k2.detach()).sum(dim=-1, keepdim=True))).mean()

class SimDis_Model(Module):
    def __init__(self, args, param_momentum, total_iters):
        super(SimDis_Model, self).__init__()
        self.total_iters = total_iters
        self.param_momentum = param_momentum
        self.current_train_iter = 0
        self.args = args

        if args.syncBN: bn = "torchsync"
        else: bn = "vanilla"
        if args.rank == 0: print(bn, 'Batch Normalization is used!')

        # customized vanilla torchsync  mbn
        if args.model_s is not None:
            width=1
            
            self.student = resnet.load_resnet(args.model_s)(
                low_dim=256, width=width, hidden_dim=4096, MLP="byol", CLS=False, bn=bn, predictors=args.model_s.split('.')[1:])
            
            self.student_ema = resnet.load_resnet(args.model_s)(
                low_dim=256, width=width, hidden_dim=4096, MLP="byol", CLS=False, bn=bn, predictors=None)

            #print(self.student)
            for p in self.student_ema.parameters():
                p.requires_grad = False

            self.update_stu(m=0)

        if args.model_t is not None:
            width=1
                
            self.teacher = resnet.load_resnet(args.model_t)(
                low_dim=256, width=width, hidden_dim=4096, MLP="byol", CLS=False, bn="torchsync", predictors=args.model_t.split('.')[1:])
            
            self.teacher_ema = resnet.load_resnet(args.model_t)(
                low_dim=256, width=width, hidden_dim=4096, MLP="byol", CLS=False, bn="torchsync", predictors=None)
            
            for p in self.teacher_ema.parameters():
                p.requires_grad = False

            if args.offline:
                for p in self.teacher.parameters():
                    p.requires_grad = False
                    
                state = torch.load(args.offline_resume, map_location='cpu')
                state_teacher = {}
                state_teacher_ema = {}
                for k, v in state['model'].items():
                    if 'pred_sema_dv' in k: k = k.replace("pred_sema_dv", "pred_ttemadv")
                    
                    if 'student_ema' in k:
                        state_teacher_ema[k.replace("module.student_ema.", "")] = v
                    elif 'student' in k:
                        state_teacher[k.replace("module.student.", "")] = v

                self.teacher.load_state_dict(state_teacher)
                self.teacher_ema.load_state_dict(state_teacher_ema)
                del state
                del state_teacher
                del state_teacher_ema

                if args.rank == 0: print('The teacher model is loaded with pre-trained weights for Offline!')

            if not args.offline: self.update_tea(m=0)
        

    @torch.no_grad()
    def update_stu(self, m):
        for p1, p2 in zip(self.student.parameters(), self.student_ema.parameters()):
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data

    @torch.no_grad()
    def update_tea(self, m):
        for p1, p2 in zip(self.teacher.parameters(), self.teacher_ema.parameters()):
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data
            

    def get_param_momentum(self):
        return 1.0 - (1.0 - self.param_momentum) * (
            (math.cos(math.pi * self.current_train_iter / self.total_iters) + 1) * 0.5
        )

    def forward(self, inps, update_param=True):
        return self.student(inps, feat=True)





        
