# MIT License
#
# Copyright (c) 2022 Intelligent Systems Lab Org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# File author: Shariq Farooq Bhat

import os
import glob
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data.distributed
from pprint import pprint
import argparse

from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import DepthDataLoader

# Set environment variables.
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"
# Disable wandb
os.environ["WANDB_MODE"] = "disabled"


def fix_random_seed(seed: int):
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    from zoedepth.models.model_io import load_wts

    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")
        checkpoint = matches[0]
    else:
        return model
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model


def main_worker(gpu, ngpus_per_node, config):
    try:
        # Set the device to GPU or CPU based on availability.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fix seed for reproducibility.
        seed = config.seed if "seed" in config and config.seed else 43
        fix_random_seed(seed)

        # Set up model and move it to the device.
        model = build_model(config)
        model = model.to(device)

        # Load model weights.
        model = load_ckpt(config, model)

        # Print model parameters.
        total_params = f"{round(count_parameters(model) / 1e6, 2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}")

        # Data loaders for training and evaluation.
        train_loader = DepthDataLoader(config, "train").data
        test_loader = DepthDataLoader(config, "online_eval").data

        # Initialize trainer (this trainer now assumes no mask is used).
        trainer = get_trainer(config)(
            config, model, train_loader, test_loader, device=device
        )

        # Start training.
        trainer.train()

    finally:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    # Uncomment and set the start method if needed.
    # mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="synunet")
    parser.add_argument("-d", "--dataset", type=str, default="nyu")
    parser.add_argument("--trainer", type=str, default=None)

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    # If using shared dict for distributed training.
    if config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    config.shared_dict = shared_dict

    config.batch_size = config.bs
    config.mode = "train"
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ["SLURM_JOB_NODELIST"].replace("[", "").replace("]", "")
        nodes = node_str.split(",")
        config.world_size = len(nodes)
        config.rank = int(os.environ["SLURM_PROCID"])
    except KeyError:
        # Not using SLURM.
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:
        print(config.rank)
        port = np.random.randint(15000, 15025)
        config.dist_url = f"tcp://{nodes[0]}:{port}"
        print(config.dist_url)
        config.dist_backend = "nccl"
        config.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)
