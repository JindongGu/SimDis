# encoding: utf-8

from SimDis.exps.arxiv.linear_eval_exp import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self, args):
        super(Exp, self).__init__(args)

        # ----------------------------- byol setting ------------------------------- #
        self.basic_lr_per_img = args.basic_lr / 256.0
        self.max_epoch = 80
        self.scheduler = "cos"  # "multistep"
        self.milestones = None
        self.save_folder_prefix = "simdis_linear_"
        
