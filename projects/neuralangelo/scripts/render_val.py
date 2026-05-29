'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import argparse
import os
import sys

sys.path.append(os.getcwd())

from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments  # noqa: E402
from imaginaire.utils.distributed import init_dist, get_world_size, is_master, master_only_print as print  # noqa: E402
from imaginaire.utils.gpu_affinity import set_affinity  # noqa: E402
from imaginaire.trainers.utils.get_trainer import get_trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Render validation outputs to videos")
    parser.add_argument("--config", required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output_dir", required=True, help="Directory to store rendered videos.")
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument("--subset", type=int, default=None,
                        help="Optional number of validation frames to render. Use 0 or a negative value for all.")
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    if args.subset is not None:
        cfg.data.val.subset = None if args.subset <= 0 else args.subset

    if not args.single_gpu:
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Running validation render with {get_world_size()} GPUs.")

    cfg.logdir = ''
    trainer = get_trainer(cfg, is_inference=True, seed=0)
    trainer.set_data_loader(cfg, split="val")
    trainer.checkpointer.load(args.checkpoint, load_opt=False, load_sch=False)
    trainer.model.eval()

    trainer.current_iteration = trainer.checkpointer.eval_iteration
    if cfg.model.object.sdf.encoding.coarse2fine.enabled:
        trainer.model_module.neural_sdf.set_active_levels(trainer.current_iteration)
        if cfg.model.object.sdf.gradient.mode == "numerical":
            trainer.model_module.neural_sdf.set_normal_epsilon()

    data_all = trainer.test(trainer.eval_data_loader, mode="val", show_pbar=True)
    if is_master():
        os.makedirs(args.output_dir, exist_ok=True)
        trainer.dump_test_results(data_all, args.output_dir)
        print(f"Saved render videos to {args.output_dir}")


if __name__ == "__main__":
    main()
