import os
import sys
import argparse
import time
import random
import warnings
import subprocess
import importlib
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed


from torch.nn.parallel import DistributedDataParallel as DDP
from SimDis.utils.log import setup_logger
from SimDis.utils import adjust_learning_rate_iter, save_checkpoint, parse_devices, AvgMeter, load_checkpoint
from SimDis.utils.torch_dist import configure_nccl, synchronize


def cleanup():
    dist.destroy_process_group()

def main():
    file_name = os.path.join(exp.output_dir, exp.exp_name)
    if args.rank == 0:
        if not os.path.exists(file_name):
            os.mkdir(file_name)

    logger = setup_logger(file_name, distributed_rank=args.local_rank, filename="train_log.txt", mode="a")
    logger.info("gpuid: {}, args: {}".format(args.local_rank, args))

    data_loader = exp.get_data_loader(batch_size=args.batchsize, is_distributed=args.nr_gpu > 1, if_transformer=False)
    train_loader = data_loader["train"]
    eval_loader = data_loader["eval"]

    model = exp.get_model().to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    optimizer = exp.get_optimizer(model.module, args.batchsize)

    start_epoch, model, optimizer = load_checkpoint(args, file_name, model, optimizer, args.linear, args.tea)

    if args.eval_method == 'knn':
        exp.eval_knn()
        return
    elif args.eval_method == 'ca':
        exp.eval_ca()
        return
    
    cudnn.benchmark = True

    if args.eval:
        model.train(False)
        exp.run_eval(model, eval_loader)
        return

    # -----------------------------start training-----------------------------#
    model.train()
    ITERS_PER_EPOCH = len(train_loader)
    if args.rank == 0:
        logger.info("Training start...")
        logger.info("Here is the logging file"+str(file_name))
        # logger.info(str(model))

    args.lr = exp.basic_lr_per_img * args.batchsize * args.nr_gpu * args.word_size
    args.warmup_epochs = exp.warmup_epochs
    args.total_epochs = exp.max_epoch
    iter_count = (start_epoch-1) * ITERS_PER_EPOCH
    model.module.current_train_iter = iter_count

    for epoch in range(start_epoch, args.total_epochs+1):
        if args.nr_gpu > 1:
            train_loader.sampler.set_epoch(epoch)
        batch_time_meter = AvgMeter()

        for i, (inps, target) in enumerate(train_loader):
            iter_count += 1
            iter_start_time = time.time()

            for indx in range(len(inps)):
                inps[indx] = inps[indx].to(device, non_blocking=True)

            data_time = time.time() - iter_start_time
            if args.linear: loss = model(inps, target)
            else: loss = model(inps, update_param=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = adjust_learning_rate_iter(optimizer, iter_count, args, ITERS_PER_EPOCH)
            batch_time_meter.update(time.time() - iter_start_time)

            if args.rank == 0 and (i + 1) % exp.print_interval == 0:
                remain_time = (ITERS_PER_EPOCH * exp.max_epoch - iter_count) * batch_time_meter.avg
                t_m, t_s = divmod(remain_time, 60)
                t_h, t_m = divmod(t_m, 60)
                t_d, t_h = divmod(t_h, 24)
                remain_time = "{}d.{:02d}h.{:02d}m".format(int(t_d), int(t_h), int(t_m))

                logger.info(
                    "[{}/{}], remain:{}, It:[{}/{}], Data-Time:{:.3f}, LR:{:.4f}, Loss:{:.2f}".format(
                        epoch, args.total_epochs, remain_time, i + 1, ITERS_PER_EPOCH, data_time, lr, loss
                    )
                )
                
        if args.linear:
            model.train(False)
            exp.run_eval(model, eval_loader)
            model.train(True)

        if args.rank == 0:
            logger.info(
                "Train-Epoch: [{}/{}], LR: {:.4f}, Con-Loss: {:.2f}".format(epoch, args.total_epochs, lr, loss)
            )

            save_checkpoint(
                {"start_epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                False,
                file_name,
                "last_epoch",
                args.linear,
            )

            save_checkpoint(
                    {"start_epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    False,
                    file_name,
                    str(epoch),
                    args.linear,
            )

    if args.rank == 0:
        print("Pre-training of experiment: {} is done.".format(args.exp_file))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SimDis")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    # optimization
    parser.add_argument("--scheduler", type=str, default="warmcos",
                        choices=["warmcos", "cos", "linear", "multistep", "step"], help="type of scheduler")

    # distributed
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--word_size", type=int, default=1)
    parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")

    # exp setting
    parser.add_argument("-n", "--n_views", type=int, default=2)
    parser.add_argument("-b", "--batchsize", type=int, default=64, help="batch size")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--basic_lr", type=float, default=0.3)
    
    parser.add_argument('--method', type=str, default='BYOL', help='choose method')
    parser.add_argument('--optimizer', type=str, default='LARS', help='SGD, LARS')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--model_s', type=str, default=None)
    parser.add_argument('--model_t', type=str, default=None)
    parser.add_argument('--ema_moment', type=float, default=0.99, help='the moment to update target network')

    parser.add_argument("--offline", action='store_true')
    parser.add_argument('--offline_resume', type=str, default='')
    
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--tea", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument('--eval_method', type=str, default=None, help='knn, ca')
    
    args = parser.parse_args()
    args.nr_gpu = torch.cuda.device_count()
    args.rank = args.rank*args.nr_gpu + args.local_rank
    
    if args.linear:
        args.batchsize = 256
        args.basic_lr = 0.001
    else:
        args.basic_lr = 0.3

    if args.local_rank == 0:
        print("V1 Using", torch.cuda.device_count(), "GPUs per node!")

    from SimDis.exps.arxiv import base_exp_simDis
    sys.path.insert(0, os.path.dirname(args.exp_file))
    current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])
    exp = current_exp.Exp(args)

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    print("Rank {} initialization finished.".format(args.rank))
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    
    main()

    if args.distributed:
        cleanup()
