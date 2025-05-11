import json
import logging
import math
import time
import torch
from training.misc import is_main_process
from open_clip import get_cast_dtype
from .distributed import is_master
from .zero_shot import multi_gpu_sync, zero_shot_eval
from .precision import get_autocast
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def format_time(seconds):
    """将秒数转换为小时、分钟和秒的格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

@torch.no_grad()
def student_teacher_ensemble(student, teacher, alpha=0.5):
    target_state_dict = {}
    for k, v in student.items():
        target_state_dict[k] = v * alpha + teacher[k] * (1.0 - alpha)
    return target_state_dict


def train_one_epoch(model, teacher_model, vfm_model, method, data, epoch, optimizer, scaler, scheduler, writer, args):
    autocast = get_autocast(args.precision)
    model.train()
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    epoch_start_time = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum
        if not args.skip_scheduler:
            scheduler(step)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        assert args.accum_freq == 1, "accum freq disabled"
        with autocast():
            losses, batch_size = method(batch, model, teacher_model, vfm_model, args)
            total_loss = sum(losses.values())
            losses["loss"] = total_loss
        backward(total_loss, scaler)
        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        # compute the epoch remaining time
        elapsed_time = time.time() - epoch_start_time
        avg_iteration_time = elapsed_time / batch_count
        remaining_iterations = num_batches_per_epoch - batch_count
        estimated_remaining_time = avg_iteration_time * remaining_iterations
        formatted_eta = format_time(estimated_remaining_time)
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            # batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"ETA: {formatted_eta} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:6f} "+ loss_log
            )
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    

def evaluate(model, data, epoch, args):
    metrics = {}
    model.eval()
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    if not is_master(args):
        return {}
    metrics.update(zero_shot_metrics)
    if not metrics:
        return metrics

    keys = ''.join([f"{k}, " for k in metrics.keys() if 'all' in k])[:-2]
    values = ''.join([f'{round(v, 4):.4f}, ' for k, v in metrics.items() if 'all' in k])[:-2]

    logging.info(
        f"Eval Epoch: {epoch-1}. "
        + f"{keys}: {values}."
    )
    # TODO save the results as plots
    logging.info(metrics)

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.json"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics
