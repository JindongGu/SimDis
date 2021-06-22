
# SimDis: Simple Distillation Baselines for Improving Small Self-supervised Models

[To Update]

## Offline Distillation Baseline
### Step 1. Train Teacher
```
python -m torch.distributed.launch --nproc_per_node=8	--nnodes=$1 --node_rank=$2 SimDis/tools/run_SimDis.py \
--distributed -d 0-7 --rank $2 --word_size $1  \
--exp_file SimDis/exps/arxiv/base_exp_simDis.py \
--optimizer LARS --syncBN --epochs 1000 \ 
--method SimDis_teacher \
--model_t resnet50.ttemadv
```
### Step 2. Train Student with offline distillation
```
python -m torch.distributed.launch --nproc_per_node=8	--nnodes=$1 --node_rank=$2 SimDis/tools/run_SimDis.py \
	--distributed -d 0-7 --rank $2 --word_size $1  \
  --exp_file SimDis/exps/arxiv/base_exp_simDis.py \
	--optimizer LARS --syncBN --epochs 1000 \ 
  --method SimDis_off \
  --model_s resnet18.s_dv.sema_sv.sema_dv.t_sv.t_dv.tema_sv.tema_dv \
	--model_t resnet50.ttemadv \
  --offline \
  --offline_resume teacher_ckpt.pth.tar
```

## Online Distillation Baseline
### Train Teacher and Student simultaneously
```
python -m torch.distributed.launch --nproc_per_node=8	--nnodes=$1 --node_rank=$2 SimDis/tools/run_SimDis.py \
	--distributed -d 0-7 --rank $2 --word_size $1  \
  --exp_file SimDis/exps/arxiv/base_exp_simDis.py \
	--optimizer LARS --syncBN --epochs 1000 \
  --method SimDis_on \
  --model_s resnet18.s_dv.sema_sv.sema_dv.t_sv.t_dv.tema_sv.tema_dv \
	--model_t resnet50.ttemadv
```

**Note:** the choice for the argument 'model_s':
2v: resnet18.sema_dv.tema_dv
7v: resnet18.s_dv.sema_sv.sema_dv.t_sv.t_dv.tema_sv.tema_dv

## Linear Evaluation
```
python -m torch.distributed.launch --nproc_per_node=8	--nnodes=$1 --node_rank=$2 SimDis/tools/run_SimDis.py \
	--distributed -d 0-7 --rank $2 --word_size $1  \
  	--exp_file SimDis/exps/arxiv/linear_eval_exp_simDis.py \
	--optimizer LARS --syncBN \
  --method SimDis_linear \
  --model_s resnet18.s_dv.sema_sv.sema_dv.t_sv.t_dv.tema_sv.tema_dv \
	--linear \
```

Contact: jindong.gu@outlook.com

Acknowledgements and Reference: 
Our Code is based on the following git repo: https://github.com/zengarden/momentum2-teacher

