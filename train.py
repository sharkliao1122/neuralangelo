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

import imaginaire.config
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.distributed import init_dist, get_world_size, master_only_print as print, is_master
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.trainers.utils.logging import init_logging
from imaginaire.trainers.utils.get_trainer import get_trainer
from imaginaire.utils.set_random_seed import set_random_seed
from projects.neuralangelo.utils.command_logger import record_training_command
from projects.neuralangelo.utils.loss_curve import plot_loss_curves
from projects.neuralangelo.utils.total_time import run_with_total_time


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.', default=None)
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--show_pbar', action='store_true')
    parser.add_argument('--total_time', action='store_true',
                        help='Print total wall-clock training time when training finishes.')
    parser.add_argument('--record_command', action='store_true',
                        help='Record this training command to the log directory.')
    parser.add_argument('--plot_loss_curve', action='store_true',
                        help='Plot loss curves from the local loss history after training finishes.')
    parser.add_argument('--render_val_after_train', action='store_true',
                        help='Render validation outputs after training finishes.')
    parser.add_argument('--render_val_subset', type=int, default=None,
                        help='Optional number of validation frames to render after training. Use 0 or a negative value for all.')
    parser.add_argument('--render_val_output_dir', default=None,
                        help='Optional output directory for post-training renders. Defaults to <logdir>/renders_final.')
    parser.add_argument('--render_val_save_frames_only', action='store_true',
                        help='Save PNG frames instead of MP4 videos for post-training renders.')
    parser.add_argument('--wandb', action='store_true', help="Enable using Weights & Biases as the logger")
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--resume', action='store_true')
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def render_validation_outputs(trainer, cfg, args):
    if args.render_val_subset is not None:
        cfg.data.val.subset = None if args.render_val_subset <= 0 else args.render_val_subset
        trainer.set_data_loader(cfg, split="val")

    output_dir = args.render_val_output_dir or os.path.join(cfg.logdir, "renders_final")
    os.makedirs(output_dir, exist_ok=True)
    data_all = trainer.test(trainer.eval_data_loader, mode="val", show_pbar=args.show_pbar)
    if is_master():
        trainer.dump_test_results(
            data_all,
            output_dir,
            save_frames_only=args.render_val_save_frames_only,
        )
        render_type = "frames" if args.render_val_save_frames_only else "videos"
        print(f"Saved post-training validation render {render_type} to {output_dir}")


def main():
    args, cfg_cmd = parse_args()
    set_affinity(args.local_rank)
    cfg = Config(args.config)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)

    # If args.single_gpu is set to True, we will disable distributed data parallel.
    if not args.single_gpu:
        # this disables nccl timeout
        os.environ["NCLL_BLOCKING_WAIT"] = "0"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank, rank=-1, world_size=-1)
    print(f"Training with {get_world_size()} GPUs.")

    # set random seed by rank
    set_random_seed(args.seed, by_rank=True)

    # Global arguments.
    imaginaire.config.DEBUG = args.debug

    # Create log directory for storing training results.
    cfg.logdir = init_logging(args.config, args.logdir, makedir=True)

    # Print and save final config
    if is_master():
        cfg.print_config()
        cfg.save_config(cfg.logdir)
        if args.record_command:
            record_training_command(
                cfg.logdir,
                argv=sys.argv,
                metadata={
                    "config": args.config,
                    "logdir": cfg.logdir,
                    "checkpoint": args.checkpoint,
                    "resume": args.resume,
                    "seed": args.seed,
                },
            )

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    trainer = get_trainer(cfg, is_inference=False, seed=args.seed)
    trainer.set_data_loader(cfg, split="train")
    trainer.set_data_loader(cfg, split="val")
    trainer.checkpointer.load(args.checkpoint, args.resume, load_sch=True, load_opt=True)

    # Initialize Wandb.
    trainer.init_wandb(cfg,
                       project=args.wandb_name,
                       mode="disabled" if args.debug or not args.wandb else "online",
                       resume=args.resume,
                       use_group=True)

    trainer.mode = 'train'
    # Start training.
    run_with_total_time(
        args.total_time,
        trainer.train,
        cfg,
        trainer.train_data_loader,
        single_gpu=args.single_gpu,
        profile=args.profile,
        show_pbar=args.show_pbar,
        output_path=os.path.join(cfg.logdir, "total_time.txt"),
    )

    if args.render_val_after_train:
        render_validation_outputs(trainer, cfg, args)

    # Finalize training.
    trainer.finalize(cfg)
    if args.plot_loss_curve and is_master():
        try:
            output_paths = plot_loss_curves(cfg.logdir)
            print(f"Saved {len(output_paths)} loss curve plot(s) to {os.path.join(cfg.logdir, 'loss_curves')}.")
        except Exception as error:
            print(f"Warning: failed to plot loss curves: {error}")


if __name__ == "__main__":
    main()
