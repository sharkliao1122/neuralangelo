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

import os

import torch
import torch.nn.functional as torch_F
import wandb
import skvideo.io
from torchvision.transforms import functional as torchvision_F

from imaginaire.utils.distributed import master_only
from imaginaire.utils.visualization import wandb_image, preprocess_image
from projects.nerf.trainers.base import BaseTrainer
from projects.neuralangelo.utils.misc import get_scheduler, eikonal_loss, curvature_loss


class Trainer(BaseTrainer):

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.metrics = dict()
        self.warm_up_end = cfg.optim.sched.warm_up_end
        self.cfg_gradient = cfg.model.object.sdf.gradient
        if cfg.model.object.sdf.encoding.type == "hashgrid" and cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.c2f_step = cfg.model.object.sdf.encoding.coarse2fine.step
            self.model.module.neural_sdf.warm_up_end = self.warm_up_end

    def _init_loss(self, cfg):
        self.criteria["render"] = torch.nn.L1Loss()

    def setup_scheduler(self, cfg, optim):
        return get_scheduler(cfg.optim, optim)

    def _compute_loss(self, data, mode=None):
        if mode == "train":
            # Compute loss only on randomly sampled rays.
            self.losses["render"] = self.criteria["render"](data["rgb"], data["image_sampled"]) * 3  # FIXME:sumRGB?!
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            if "eikonal" in self.weights.keys():
                self.losses["eikonal"] = eikonal_loss(data["gradients"], outside=data["outside"])
            if "curvature" in self.weights:
                self.losses["curvature"] = curvature_loss(data["hessians"], outside=data["outside"])
        else:
            # Compute loss on the entire image.
            self.losses["render"] = self.criteria["render"](data["rgb_map"], data["image"])
            self.metrics["psnr"] = -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()

    def get_curvature_weight(self, current_iteration, init_weight):
        if "curvature" in self.weights:
            if current_iteration <= self.warm_up_end:
                self.weights["curvature"] = current_iteration / self.warm_up_end * init_weight
            else:
                model = self.model_module
                decay_factor = model.neural_sdf.growth_rate ** (model.neural_sdf.anneal_levels - 1)
                self.weights["curvature"] = init_weight / decay_factor

    def _start_of_iteration(self, data, current_iteration):
        model = self.model_module
        self.progress = model.progress = current_iteration / self.cfg.max_iter
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            model.neural_sdf.set_active_levels(current_iteration)
            if self.cfg_gradient.mode == "numerical":
                model.neural_sdf.set_normal_epsilon()
                self.get_curvature_weight(current_iteration, self.cfg.trainer.loss_weight.curvature)
        elif self.cfg_gradient.mode == "numerical":
            model.neural_sdf.set_normal_epsilon()

        return super()._start_of_iteration(data, current_iteration)

    @master_only
    def log_wandb_scalars(self, data, mode=None):
        super().log_wandb_scalars(data, mode=mode)
        scalars = {
            f"{mode}/PSNR": self.metrics["psnr"].detach(),
            f"{mode}/s-var": self.model_module.s_var.item(),
        }
        if "curvature" in self.weights:
            scalars[f"{mode}/curvature_weight"] = self.weights["curvature"]
        if "eikonal" in self.weights:
            scalars[f"{mode}/eikonal_weight"] = self.weights["eikonal"]
        if mode == "train" and self.cfg_gradient.mode == "numerical":
            scalars[f"{mode}/epsilon"] = self.model.module.neural_sdf.normal_eps
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            scalars[f"{mode}/active_levels"] = self.model.module.neural_sdf.active_levels
        wandb.log(scalars, step=self.current_iteration)

    @master_only
    def log_wandb_images(self, data, mode=None, max_samples=None):
        images = {"iteration": self.current_iteration, "epoch": self.current_epoch}
        if mode == "val":
            images_error = (data["rgb_map"] - data["image"]).abs()
            images.update({
                f"{mode}/vis/rgb_target": wandb_image(data["image"]),
                f"{mode}/vis/rgb_render": wandb_image(data["rgb_map"]),
                f"{mode}/vis/rgb_error": wandb_image(images_error),
                f"{mode}/vis/normal": wandb_image(data["normal_map"], from_range=(-1, 1)),
                f"{mode}/vis/inv_depth": wandb_image(1 / (data["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale),
                f"{mode}/vis/opacity": wandb_image(data["opacity_map"]),
            })
        wandb.log(images, step=self.current_iteration)

    def dump_test_results(self, data_all, output_dir):
        results = dict(
            rgb_target=preprocess_image(data_all["image"]),
            rgb_render=preprocess_image(data_all["rgb_map"]),
            rgb_error=preprocess_image((data_all["rgb_map"] - data_all["image"]).abs()),
            normal=preprocess_image(data_all["normal_map"], from_range=(-1, 1)),
            inv_depth=preprocess_image(1 / (data_all["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale),
            opacity=preprocess_image(data_all["opacity_map"]),
        )
        inputdict, outputdict = self._get_ffmpeg_dicts()
        try:
            for key, image_list in results.items():
                print(f"writing video ({key})...")
                video_fname = f"{output_dir}/{key}.mp4"
                video_writer = skvideo.io.FFmpegWriter(video_fname, inputdict=inputdict, outputdict=outputdict)
                for image in image_list:
                    image = (image * 255).byte().permute(1, 2, 0).numpy()
                    video_writer.writeFrame(image)
                video_writer.close()
        except AssertionError as exc:
            if "FFmpeg" not in str(exc):
                raise
            print(f"FFmpeg unavailable, saving PNG frames instead: {exc}")
            self._dump_test_frames(results, output_dir)

    def _dump_test_frames(self, results, output_dir):
        frames_root = os.path.join(output_dir, "frames")
        for key, image_list in results.items():
            key_dir = os.path.join(frames_root, key)
            os.makedirs(key_dir, exist_ok=True)
            print(f"writing frames ({key})...")
            for idx, image in enumerate(image_list):
                image_path = os.path.join(key_dir, f"{idx:04d}.png")
                torchvision_F.to_pil_image((image * 255).byte()).save(image_path)

    def _get_ffmpeg_dicts(self):
        inputdict = {"-r": str(30)}
        outputdict = {"-crf": str(10), "-pix_fmt": "yuv420p"}
        return inputdict, outputdict

    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        self.progress = self.model_module.progress = self.current_iteration / self.cfg.max_iter
        super().train(cfg, data_loader, single_gpu, profile, show_pbar)
