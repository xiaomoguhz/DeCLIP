import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from tools.k_means import run_kmeans
from tools.precompute_knns import run_knns
from tools.segmentation import run_seg
from training.misc import is_main_process
from training.declip import DeCLIP
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from open_clip import create_model_and_transforms, get_tokenizer, create_model
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate, student_teacher_ensemble
from training.file_utils import pt_load
from .utils import freeze_parameters, build_vfm
from torch.utils.tensorboard import SummaryWriter

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    log_base_path = os.path.join(args.logs, args.name)
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=log_base_path)
    else:
        writer = None
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print("WARNING, Experiment already exists.")
    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    student_model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
        det_image_size=args.det_image_size,
        dataset_type=args.dataset_type,
        args=args
    )
    random_seed(args.seed, args.rank)

    teacher_model = create_model(
        args.model, 
        args.pretrained,
        device=device,
        precision=args.precision,
        output_dict=True,
        cache_dir=args.cache_dir).to(args.device)
    for p in teacher_model.parameters():
        p.requires_grad = False
 
    if hasattr(student_model, 'visual'):
        args.input_size = student_model.visual.image_size
    elif hasattr(student_model, 'vision_model'):
        args.input_size = student_model.vision_model.image_size
    else:
        raise ValueError("student_model must have either 'visual' or 'vision_model' attribute")

    if args.lock_image:
        student_model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats,)
    if args.grad_checkpointing:
        student_model.set_grad_checkpointing()
    student_model = freeze_parameters(student_model,args)
    if args.use_vfm:
        vfm_model = build_vfm(args.use_vfm)
        vfm_model = vfm_model.to(args.device)
    else:
        vfm_model=None
        
    method = DeCLIP()
    student_model_without_ddp = student_model
    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(student_model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    
    if args.distributed:
        if args.use_bn_sync:
            if args.use_vfm:
                student_model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
            else:
                student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
        ddp_args = {} 
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[device], **ddp_args)
        student_model_without_ddp=student_model.module

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data:
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)
        named_parameters = list(student_model_without_ddp.named_parameters()) 
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        sd = checkpoint["state_dict"]
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            student_model_without_ddp.load_state_dict(sd)
            if args.dataset_type == "froster":
                teacher_model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if is_main_process():
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            student_model_without_ddp.load_state_dict(sd)
            if args.dataset_type == "froster":
                teacher_model.load_state_dict(checkpoint)
            if is_main_process():
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'
    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const": 
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)
    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    if 'train' not in data or args.eval or args.precompute_knn:
        del teacher_model
        if args.k_means:
            del vfm_model
            run_kmeans(student_model_without_ddp,data,args)
        elif args.run_seg:
            del vfm_model
            run_seg(student_model_without_ddp,data,args)
        elif args.precompute_knn:
            del student_model
            run_knns(vfm_model,data,args)
        else:
            del vfm_model
            evaluate(student_model_without_ddp, data, start_epoch, args)
        return
    if not args.skip_first_eval:
        if is_main_process():
            logging.info('Evaluate before training')
        evaluate(student_model_without_ddp, data, start_epoch, args)
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        train_one_epoch(student_model,
                        teacher_model,
                        vfm_model,
                        method,
                        data,epoch,optimizer, scaler, scheduler, writer, args)
        completed_epoch = epoch + 1
        student_state_dict = student_model_without_ddp.state_dict()
        if args.alpha < 1.0:
            teacher_state_dict = teacher_model.state_dict()
            target_state_dict = student_teacher_ensemble(student_state_dict, teacher_state_dict, args.alpha)
        else:
            target_state_dict = student_state_dict
        if is_master(args):
            # Saving checkpoints.
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": target_state_dict,
                "optimizer": optimizer.state_dict()}
            
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()
            if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ): 
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

        if completed_epoch % args.zeroshot_frequency == 0:
            test_model = create_model(
                args.model,
                args.pretrained,
                device=device,
                precision=args.precision,
                output_dict=True,
                cache_dir=args.cache_dir)
            test_model.load_state_dict(target_state_dict)
            evaluate(test_model, data, completed_epoch, args)
            del test_model
    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])
