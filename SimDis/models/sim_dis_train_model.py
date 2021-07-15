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

            for p in self.student_ema.parameters():
                p.requires_grad = False

            self.update_stu(m=0)

        if args.model_t is not None:
            width=1
                
            self.teacher = resnet.load_resnet(args.model_t)(
                low_dim=256, width=width, hidden_dim=4096, MLP="byol", CLS=False, bn=bn, predictors=args.model_t.split('.')[1:])
            
            self.teacher_ema = resnet.load_resnet(args.model_t)(
                low_dim=256, width=width, hidden_dim=4096, MLP="byol", CLS=False, bn=bn, predictors=None)
            
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
        if update_param:
            current_param_momentum = self.get_param_momentum()
            self.update_stu(current_param_momentum)
            if (self.args.model_t is not None) and (not self.args.offline): self.update_tea(current_param_momentum)

        loss_s = 0.
        num_view_s = 0.
        loss_t = 0.
        num_view_t = 0.
        args = self.args

        x1 = x1_ema = x1_t = x1_t_ema = inps[0]
        x2 = x2_ema = x2_t = x2_t_ema = inps[1]
        
        if args.model_s is not None:
            fs1 = self.student(x1)
            fs2 = self.student(x2)

            fs1_norm = F.normalize(fs1, dim=1)
            fs2_norm = F.normalize(fs2, dim=1)

            with torch.no_grad():
                fs1_ema = self.student_ema(x1_ema)
                fs2_ema = self.student_ema(x2_ema)

        if args.model_t is not None:
            if (not args.offline) or ('t_' in args.model_s):
                ft1 = self.teacher(x1_t)
                ft2 = self.teacher(x2_t)

                ft1_norm = F.normalize(ft1, dim=1)
                ft2_norm = F.normalize(ft2, dim=1)

            if 'tema' in args.model_s:
                with torch.no_grad():
                    ft1_ema = self.teacher_ema(x1_t_ema)
                    ft2_ema = self.teacher_ema(x2_t_ema)

        if 's_dv' in args.model_s:
            ps1_v1 = self.student.pred_s_dv(fs1)
            ps1_v2 = self.student.pred_s_dv(fs2)
            
            loss_s += pred_feat_loss(ps1_v1, fs2_norm, ps1_v2, fs1_norm)
            num_view_s += 1.

        if 'sema_sv' in args.model_s:
            ps2_v1 = self.student.pred_sema_sv(fs1)
            ps2_v2 = self.student.pred_sema_sv(fs2)
            
            loss_s += pred_feat_loss(ps2_v1, fs1_ema, ps2_v2, fs2_ema)
            num_view_s += 1.

        if 'sema_dv' in args.model_s:
            ps3_v1 = self.student.pred_sema_dv(fs1)
            ps3_v2 = self.student.pred_sema_dv(fs2)
            
            loss_s += pred_feat_loss(ps3_v1, fs2_ema, ps3_v2, fs1_ema)
            num_view_s += 1.

        if 't_sv' in args.model_s:
            ps4_v1 = self.student.pred_t_sv(fs1)
            ps4_v2 = self.student.pred_t_sv(fs2)
            
            loss_s += pred_feat_loss(ps4_v1, ft1_norm, ps4_v2, ft2_norm)
            num_view_s += 1.

        if 't_dv' in args.model_s:
            ps5_v1 = self.student.pred_t_dv(fs1)
            ps5_v2 = self.student.pred_t_dv(fs2)
            
            loss_s += pred_feat_loss(ps5_v1, ft2_norm, ps5_v2, ft1_norm)
            num_view_s += 1.

        if 'tema_sv' in args.model_s:
            ps6_v1 = self.student.pred_tema_sv(fs1)
            ps6_v2 = self.student.pred_tema_sv(fs2)
            
            loss_s += pred_feat_loss(ps6_v1, ft1_ema, ps6_v2, ft2_ema)
            num_view_s += 1.

        if 'tema_dv' in args.model_s:
            ps7_v1 = self.student.pred_tema_dv(fs1)
            ps7_v2 = self.student.pred_tema_dv(fs2)
            
            loss_s += pred_feat_loss(ps7_v1, ft2_ema, ps7_v2, ft1_ema)
            num_view_s += 1.

        if (args.model_t is not None) and (not args.offline):
            
            if 'ttemadv' in args.model_t:
                if 'tema' not in args.model_s:
                    with torch.no_grad():
                        ft1_ema = self.teacher_ema(x1_t_ema)
                        ft2_ema = self.teacher_ema(x2_t_ema)
                    
                pt5_v1 = self.teacher.pred_ttemadv(ft1)
                pt5_v2 = self.teacher.pred_ttemadv(ft2)
                
                loss_t += pred_feat_loss(pt5_v1, ft2_ema, pt5_v2, ft1_ema)
                num_view_t += 1.
            
        if   num_view_t == 0: loss = loss_s/num_view_s 
        elif num_view_s == 0: loss = loss_t/num_view_t
        else: loss = (loss_s/num_view_s) + (loss_t/num_view_t)

        self.current_train_iter += 1
        if self.training:
            return loss











        
