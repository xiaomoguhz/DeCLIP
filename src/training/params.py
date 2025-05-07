import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=20)
    parser.add_argument(
        "--skip-first-eval",
        action="store_true",
        default=False)
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False)
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=16)
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,    # not used when alpha >=1.0
    )
    parser.add_argument(
        "--use_vfm",
        type=str,
        choices=["sam-B", "sam-L","dinov2-L","dinov2-B","dino-B-8","dino-B-16"],
        default="",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--image-crop-size",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--pre-transforms",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-split",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="checkpoints",
    )
    parser.add_argument(
        "--loss_content_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--loss_context_weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--det-image-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--train-image-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--image-ave-pool",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--train-image-root",
        type=str,
        default="data/coco/val2017",
    )
    parser.add_argument(
        "--train-ceph-root",
        type=str,
        default="",
    )
    parser.add_argument(
        "--val-image-root",
        type=str,
        default="data/coco/val2017",
    )
    parser.add_argument(
        "--val-segm-root",
        type=str,
        default="data/coco/annotations/panoptic_val2017",
    )
    parser.add_argument(
        "--train-segm-root",
        type=str,
        default="data/coco/annotations/panoptic_val2017",
    )
    parser.add_argument(
        "--embed-path",
        type=str,
        default="metadata/coco_clip_hand_craft_RN50.npy",
    )
    parser.add_argument(
        "--train-embed-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="",
        help="Path to file(s) with training data. When using webdataset, "
             "multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/coco/annotations/instances_val2017_100.json"
    )
    parser.add_argument(
        "--dataset-type",
        choices=['proposals_distill', "region_clip", "grid_distill","knn_grid_distill","coco_caption",],
        default="grid_distill",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--test-type",
        choices=['coco_panoptic'],
        default="coco_panoptic",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        default=False,
        help="Where to store tensorboard logs.",
    )
    parser.add_argument(
        "--precompute-knn",
        action="store_true",
        default=False,
        help="Where to precompute-knn.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--mode",
        choices=["qq", "kk", "vv","csa", "qq_vfm_distill","kk_vfm_distill",
                 "vv_vfm_distill","csa_vfm_distill","all_vfm_distill","maskclip","vanilla","sanity_check"],
        default="qq_vfm_distill",
        help="Choosing an attention mode for training and inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=3,           # freeze at 2
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=True,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--k-means",
        default=False,
        action='store_true',
        help="run k-means on evaluation set",
    )
    parser.add_argument(
        "--run-seg",
        default=False,
        action='store_true',
        help="run open-vocabulary segmentation on evaluation set",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
