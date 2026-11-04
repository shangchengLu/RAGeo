"""
train_balanced_fov_sscl_cvusa.py

Standalone CVUSA training entrypoint for Balanced-FoV SSCL and optional
Residual-aware FN-LCR-style ranking distillation.

It does NOT modify the original ConGeo/Sample4Geo project files.

Core design:
  1) Keep original ConGeo global objective unchanged:
       L(q1,r1) + 0.5 L(q1,q2_180) + 0.5 L(r1,r2) + 0.25 L(r1,q2_180)
  2) Keep the validated SSCL branch on q2_180-r1:
       lambda_180 * L_strip(q2_180, r1)
  3) Add a small-weight balanced-FoV auxiliary local branch:
       q_aux is sampled per epoch from {360,180,90,70};
       lambda_aux * L_strip(q_aux, r1)
  4) Optionally add residual-aware FN-LCR-style distillation:
       Only distill rows where the local residual improves the positive-vs-hard-negative margin.

Recommended first run:
  Balanced-FoV SSCL only, no distillation.

Recommended second run:
  Balanced-FoV SSCL + residual-aware distillation with beta=0.01.
"""

import argparse
import copy
import gc
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from rageo.dataset.cvusa import CVUSADatasetEval
from rageo.evaluate.cvusa_and_cvact import calc_sim, evaluate
from rageo.loss import InfoNCE
from rageo.model import TimmModel_ConGeo
from rageo.transforms import get_transforms_train_congeo, get_transforms_val
from rageo.utils import Logger, setup_system


@dataclass
class Configuration:
    dataset: str = "cvusa"
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    local_pretrain_path: str = "./pretrained/pytorch_model.bin"
    img_size: int = 384

    seed: int = 42
    epochs: int = 60
    batch_size: int = 16
    batch_size_eval: int = 16
    num_workers: int = 8
    gpu_ids: Tuple[int, ...] = (0, 1, 2, 3)
    mixed_precision: bool = True
    verbose: bool = True

    custom_sampling: bool = True
    gps_sample: bool = True
    sim_sample: bool = True
    neighbour_select: int = 64
    neighbour_range: int = 128
    gps_dict_path: str = "./datasets/CVUSA/gps_dict.pkl"

    eval_every_n_epoch: int = 4
    normalize_features: bool = True
    clip_grad: float = 100.0
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False
    label_smoothing: float = 0.1

    lr: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    lr_end: float = 1e-5

    data_folder: str =  "./datasets/CVUSA"
    prob_rotate: float = 0.75
    prob_flip: float = 0.5
    model_path: str = "./balanced_fov_sscl_cvusa"
    checkpoint_start: Optional[str] = None
    save_every_epoch: bool = True
    save_latest: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    train_fov: float = 180.0
    eval_fov: float = 90.0
    random_fov: bool = False

    # Validated SSCL branch on q2_180-r1.
    sscl180_K: int = 2
    sscl180_lambda: float = 0.25

    # Balanced-FoV auxiliary local branch.
    use_aux: bool = True
    aux_fovs: Tuple[float, ...] = (360.0, 180.0, 90.0, 70.0)
    aux_schedule: str = "cycle"  # cycle or random
    aux_K: int = 2
    aux_lambda: float = 0.05
    local_warmup_epochs: int = 5

    # Optional residual-aware distillation.
    use_distill: bool = False
    distill_on: str = "aux"  # aux, q2, both
    distill_beta: float = 0.01
    distill_warmup_epochs: int = 5
    distill_lcr_lambda: float = 3.0
    tau_teacher: float = 1.0
    tau_student: float = 1.0
    residual_weight_mode: str = "positive"  # positive or soft or all
    residual_soft_tau: float = 0.2

    run_name: str = ""


def str2bool_int(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y"}


def parse_fovs(text: str) -> Tuple[float, ...]:
    vals = []
    for s in text.replace(",", " ").split():
        if s.strip():
            vals.append(float(s))
    if not vals:
        raise ValueError("aux_fovs cannot be empty")
    return tuple(vals)


def parse_args() -> Configuration:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=Configuration.data_folder)
    parser.add_argument("--gps_dict_path", type=str, default=Configuration.gps_dict_path)
    parser.add_argument("--local_pretrain_path", type=str, default=Configuration.local_pretrain_path)
    parser.add_argument("--checkpoint_start", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=Configuration.model_path)
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--epochs", type=int, default=Configuration.epochs)
    parser.add_argument("--batch_size", type=int, default=Configuration.batch_size)
    parser.add_argument("--batch_size_eval", type=int, default=Configuration.batch_size_eval)
    parser.add_argument("--num_workers", type=int, default=Configuration.num_workers)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3")

    parser.add_argument("--lr", type=float, default=Configuration.lr)
    parser.add_argument("--scheduler", type=str, default=Configuration.scheduler, choices=["cosine", "polynomial", "constant", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=Configuration.warmup_epochs)
    parser.add_argument("--eval_every_n_epoch", type=int, default=Configuration.eval_every_n_epoch)

    parser.add_argument("--train_fov", type=float, default=Configuration.train_fov)
    parser.add_argument("--eval_fov", type=float, default=Configuration.eval_fov)

    parser.add_argument("--sscl180_K", type=int, default=Configuration.sscl180_K)
    parser.add_argument("--sscl180_lambda", type=float, default=Configuration.sscl180_lambda)
    parser.add_argument("--use_aux", type=str, default="1")
    parser.add_argument("--aux_fovs", type=str, default="360 180 90 70")
    parser.add_argument("--aux_schedule", type=str, default=Configuration.aux_schedule, choices=["cycle", "random"])
    parser.add_argument("--aux_K", type=int, default=Configuration.aux_K)
    parser.add_argument("--aux_lambda", type=float, default=Configuration.aux_lambda)
    parser.add_argument("--local_warmup_epochs", type=int, default=Configuration.local_warmup_epochs)

    parser.add_argument("--use_distill", type=str, default="0")
    parser.add_argument("--distill_on", type=str, default=Configuration.distill_on, choices=["aux", "q2", "both"])
    parser.add_argument("--distill_beta", type=float, default=Configuration.distill_beta)
    parser.add_argument("--distill_warmup_epochs", type=int, default=Configuration.distill_warmup_epochs)
    parser.add_argument("--distill_lcr_lambda", type=float, default=Configuration.distill_lcr_lambda)
    parser.add_argument("--tau_teacher", type=float, default=Configuration.tau_teacher)
    parser.add_argument("--tau_student", type=float, default=Configuration.tau_student)
    parser.add_argument("--residual_weight_mode", type=str, default=Configuration.residual_weight_mode, choices=["positive", "soft", "all"])
    parser.add_argument("--residual_soft_tau", type=float, default=Configuration.residual_soft_tau)

    parser.add_argument("--mixed_precision", type=str, default="1")
    parser.add_argument("--custom_sampling", type=str, default="1")
    parser.add_argument("--gps_sample", type=str, default="1")
    parser.add_argument("--sim_sample", type=str, default="1")
    parser.add_argument("--save_every_epoch", type=str, default="1")
    parser.add_argument("--save_latest", type=str, default="1")
    parser.add_argument("--verbose", type=str, default="1")

    args = parser.parse_args()
    cfg = Configuration()
    cfg.data_folder = args.data_folder
    cfg.gps_dict_path = args.gps_dict_path
    cfg.local_pretrain_path = args.local_pretrain_path
    cfg.checkpoint_start = args.checkpoint_start
    cfg.model_path = args.model_path
    cfg.run_name = args.run_name
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.batch_size_eval = args.batch_size_eval
    cfg.num_workers = args.num_workers
    cfg.gpu_ids = tuple(int(x) for x in args.gpu_ids.split(",") if x.strip())
    cfg.lr = args.lr
    cfg.scheduler = args.scheduler
    cfg.warmup_epochs = args.warmup_epochs
    cfg.eval_every_n_epoch = args.eval_every_n_epoch
    cfg.train_fov = args.train_fov
    cfg.eval_fov = args.eval_fov
    cfg.sscl180_K = args.sscl180_K
    cfg.sscl180_lambda = args.sscl180_lambda
    cfg.use_aux = str2bool_int(args.use_aux)
    cfg.aux_fovs = parse_fovs(args.aux_fovs)
    cfg.aux_schedule = args.aux_schedule
    cfg.aux_K = args.aux_K
    cfg.aux_lambda = args.aux_lambda
    cfg.local_warmup_epochs = args.local_warmup_epochs
    cfg.use_distill = str2bool_int(args.use_distill)
    cfg.distill_on = args.distill_on
    cfg.distill_beta = args.distill_beta
    cfg.distill_warmup_epochs = args.distill_warmup_epochs
    cfg.distill_lcr_lambda = args.distill_lcr_lambda
    cfg.tau_teacher = args.tau_teacher
    cfg.tau_student = args.tau_student
    cfg.residual_weight_mode = args.residual_weight_mode
    cfg.residual_soft_tau = args.residual_soft_tau
    cfg.mixed_precision = str2bool_int(args.mixed_precision)
    cfg.custom_sampling = str2bool_int(args.custom_sampling)
    cfg.gps_sample = str2bool_int(args.gps_sample)
    cfg.sim_sample = str2bool_int(args.sim_sample)
    cfg.save_every_epoch = str2bool_int(args.save_every_epoch)
    cfg.save_latest = str2bool_int(args.save_latest)
    cfg.verbose = str2bool_int(args.verbose)
    return cfg


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_model_state(model, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), save_path)


def load_local_timm_pretrain(model, local_pretrain_path: str):
    print("Load local pretrained weights:", local_pretrain_path)
    state_dict = torch.load(local_pretrain_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("model.", "")
        new_state_dict[k] = v
    missing, unexpected = model.model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    print("First missing:", missing[:10])
    print("First unexpected:", unexpected[:10])


def load_resume_checkpoint(model, checkpoint_path: str):
    print("Resume from checkpoint:", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Resume missing keys:", len(missing))
    print("Resume unexpected keys:", len(unexpected))
    print("First resume missing:", missing[:10])
    print("First resume unexpected:", unexpected[:10])


def check_required_files(cfg: Configuration):
    required = [
        os.path.join(cfg.data_folder, "splits", "train-19zl.csv"),
        os.path.join(cfg.data_folder, "splits", "val-19zl.csv"),
    ]
    required.append(cfg.checkpoint_start if cfg.checkpoint_start else cfg.local_pretrain_path)
    if cfg.gps_sample:
        required.append(cfg.gps_dict_path)
    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    print("Required files checked.")


class CVUSADatasetTrainConGeoBalancedAux(Dataset):
    """CVUSA ConGeo train dataset with an additional q_aux branch.

    q_aux uses one FoV for the whole epoch to keep tensor width consistent in a batch.
    Call set_aux_fov(fov) before each epoch.
    """

    def __init__(self, data_folder, transforms_query1, transforms_query2,
                 transforms_reference1, transforms_reference2, transforms_query_aux_by_fov,
                 aux_fovs: Tuple[float, ...], prob_flip=0.0, prob_rotate=0.0,
                 shuffle_batch_size=128):
        super().__init__()
        self.data_folder = data_folder
        self.transforms_query1 = transforms_query1
        self.transforms_query2 = transforms_query2
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.transforms_query_aux_by_fov = transforms_query_aux_by_fov
        self.aux_fovs = tuple(float(x) for x in aux_fovs)
        self.current_aux_fov = self.aux_fovs[0]
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.df = pd.read_csv(f"{data_folder}/splits/train-19zl.csv", header=None)
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        self.idx2pair = {p[0]: p for p in self.pairs}
        self.train_ids = [p[0] for p in self.pairs]
        self.samples = copy.deepcopy(self.train_ids)

    def set_aux_fov(self, fov: float):
        fov = float(fov)
        if fov not in self.transforms_query_aux_by_fov:
            raise ValueError(f"Aux FoV {fov} not configured. Available: {list(self.transforms_query_aux_by_fov.keys())}")
        self.current_aux_fov = fov

    def __getitem__(self, index):
        idx, sat, ground = self.idx2pair[self.samples[index]]
        query_img = cv2.imread(f"{self.data_folder}/{ground}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        query_img1 = self.transforms_query1(image=query_img)["image"]
        query_img2 = self.transforms_query2(image=query_img)["image"]
        query_aux = self.transforms_query_aux_by_fov[self.current_aux_fov](image=query_img)["image"]
        reference_img1 = self.transforms_reference1(image=reference_img)["image"]
        reference_img2 = self.transforms_reference2(image=reference_img)["image"]

        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))
            # Keep the original ConGeo implementation style: roll ground views.
            c, h, w1 = query_img1.shape
            query_img1 = torch.roll(query_img1, shifts=-(w1 // 4) * r, dims=2)
            c, h, w2 = query_img2.shape
            query_img2 = torch.roll(query_img2, shifts=-(w2 // 4) * r, dims=2)
            c, h, wa = query_aux.shape
            query_aux = torch.roll(query_aux, shifts=-(wa // 4) * r, dims=2)

        label = torch.tensor(idx, dtype=torch.long)
        aux_fov_tensor = torch.tensor(self.current_aux_fov, dtype=torch.float32)
        return query_img1, query_img2, query_aux, reference_img1, reference_img2, label, aux_fov_tensor

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        print("\nShuffle Dataset:")
        idx_pool = copy.deepcopy(self.train_ids)
        neighbour_split = neighbour_select // 2
        similarity_pool = copy.deepcopy(sim_dict) if sim_dict is not None else None
        random.shuffle(idx_pool)
        idx_epoch = set()
        idx_batch = set()
        batches = []
        current_batch = []
        break_counter = 0
        pbar = tqdm()
        while True:
            pbar.update()
            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)
                if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0
                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        near_similarity = similarity_pool[idx][:neighbour_range]
                        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                        random.shuffle(far_neighbours)
                        far_neighbours = far_neighbours[:neighbour_split]
                        for idx_near in near_neighbours + far_neighbours:
                            if len(current_batch) >= self.shuffle_batch_size:
                                break
                            if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0
                else:
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)
                    break_counter += 1
                if break_counter >= 1024:
                    break
            else:
                break
            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []
        pbar.close()
        time.sleep(0.3)
        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))


class TimmModelBalanced(TimmModel_ConGeo):
    @staticmethod
    def _to_nchw(fmap: torch.Tensor) -> torch.Tensor:
        if fmap.ndim != 4:
            raise RuntimeError(f"Expected 4D feature map, got {tuple(fmap.shape)}")
        if fmap.shape[1] < fmap.shape[-1] and fmap.shape[-1] in [512, 768, 1024, 1536]:
            fmap = fmap.permute(0, 3, 1, 2).contiguous()
        return fmap

    def forward(self, imgq1, imgq2=None, imgr1=None, imgr2=None, return_fmap: bool = False):
        if not return_fmap:
            return super().forward(imgq1, imgq2, imgr1, imgr2)
        fmap = self.model.forward_features(imgq1)
        fmap = self._to_nchw(fmap)
        desc = self.model.forward_head(fmap)
        return desc, fmap


class StripContrastiveObjective(nn.Module):
    def __init__(self, K: int = 2, eps: float = 1e-6):
        super().__init__()
        self.K = K
        self.eps = eps

    @staticmethod
    def strip_pool(fmap: torch.Tensor, K: int) -> torch.Tensor:
        B, C, H, W = fmap.shape
        K_eff = max(1, min(int(K), int(W)))
        strips = []
        for k in range(K_eff):
            w0 = k * W // K_eff
            w1 = (k + 1) * W // K_eff
            if w1 <= w0:
                w1 = min(W, w0 + 1)
            strip = fmap[:, :, :, w0:w1].mean(dim=(2, 3))
            strips.append(strip)
        strips = torch.stack(strips, dim=1)
        return F.normalize(strips, dim=-1)

    @staticmethod
    def compute_local_sim_matrix(q_strips: torch.Tensor, r_strips: torch.Tensor) -> torch.Tensor:
        sim = torch.einsum("bkc,jmc->bjkm", q_strips, r_strips)  # [B,B,Kq,Kr]
        sim_max = sim.max(dim=3).values  # [B,B,Kq]
        return sim_max.mean(dim=2)       # [B,B]

    def forward_sim(self, q_fmap: torch.Tensor, r_fmap: torch.Tensor, K: Optional[int] = None) -> torch.Tensor:
        K = self.K if K is None else K
        q_strips = self.strip_pool(q_fmap, K)
        r_strips = self.strip_pool(r_fmap, K)
        return self.compute_local_sim_matrix(q_strips, r_strips)

    @staticmethod
    def ce_from_sim(sim_matrix: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        B = sim_matrix.size(0)
        labels = torch.arange(B, dtype=torch.long, device=sim_matrix.device)
        logits = sim_matrix * logit_scale
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    @staticmethod
    def margin_stats(sim_matrix: torch.Tensor) -> Dict[str, float]:
        B = sim_matrix.size(0)
        labels = torch.arange(B, device=sim_matrix.device)
        pos = sim_matrix[labels, labels]
        neg = sim_matrix.masked_fill(torch.eye(B, device=sim_matrix.device, dtype=torch.bool), -1e4).max(dim=1).values
        margin = pos - neg
        return {
            "margin_mean": float(margin.mean().detach().cpu()),
            "margin_pos_ratio": float((margin > 0).float().mean().detach().cpu()),
        }


def row_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def residual_aware_lcr_distill(s_global: torch.Tensor, s_local: torch.Tensor,
                               lcr_lambda: float = 3.0, tau_teacher: float = 1.0,
                               tau_student: float = 1.0, weight_mode: str = "positive",
                               soft_tau: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    B = s_global.size(0)
    labels = torch.arange(B, device=s_global.device)
    mask = torch.eye(B, device=s_global.device, dtype=torch.bool)

    student_logits = row_zscore(s_global)
    with torch.no_grad():
        zg = row_zscore(s_global.detach())
        zl = row_zscore(s_local.detach())
        teacher_logits = zg + lcr_lambda * zl
        g_pos = zg[labels, labels]
        g_neg = zg.masked_fill(mask, -1e4).max(dim=1).values
        t_pos = teacher_logits[labels, labels]
        t_neg = teacher_logits.masked_fill(mask, -1e4).max(dim=1).values
        residual_gain = (t_pos - t_neg) - (g_pos - g_neg)
        if weight_mode == "all":
            weights = torch.ones_like(residual_gain)
        elif weight_mode == "positive":
            weights = (residual_gain > 0).float()
            if weights.sum() < 1:
                weights = torch.ones_like(residual_gain) * 0.0
        elif weight_mode == "soft":
            weights = torch.sigmoid(residual_gain / max(soft_tau, 1e-6))
        else:
            raise ValueError(f"Unknown residual weight mode: {weight_mode}")
        p_teacher = F.softmax(teacher_logits / tau_teacher, dim=1)

    log_p_student = F.log_softmax(student_logits / tau_student, dim=1)
    per_row_kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=1)
    denom = weights.sum().clamp_min(1.0)
    loss = (weights * per_row_kl).sum() / denom
    stats = {
        "res_gain_mean": float(residual_gain.mean().detach().cpu()),
        "res_pos_ratio": float((residual_gain > 0).float().mean().detach().cpu()),
        "res_w_mean": float(weights.mean().detach().cpu()),
    }
    return loss, stats


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def train_one_epoch(cfg: Configuration, model, dataloader, loss_function, local_obj,
                    optimizer, epoch: int, scheduler=None, scaler=None):
    model.train()
    meters = {k: AverageMeter() for k in [
        "loss", "global", "sscl180", "aux", "distill", "m180", "maux", "rgain", "rpos", "aux_fov"
    ]}
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(dataloader, total=len(dataloader)) if cfg.verbose else dataloader

    sscl180_lambda = cfg.sscl180_lambda if epoch > cfg.local_warmup_epochs else 0.0
    aux_lambda = cfg.aux_lambda if (cfg.use_aux and epoch > cfg.local_warmup_epochs) else 0.0
    distill_beta = cfg.distill_beta if (cfg.use_distill and epoch > cfg.distill_warmup_epochs) else 0.0

    for query1, query2, qaux, reference1, reference2, ids, aux_fov in bar:
        query1 = query1.to(cfg.device, non_blocking=True)
        query2 = query2.to(cfg.device, non_blocking=True)
        qaux = qaux.to(cfg.device, non_blocking=True)
        reference1 = reference1.to(cfg.device, non_blocking=True)
        reference2 = reference2.to(cfg.device, non_blocking=True)

        if scaler is not None:
            with autocast("cuda"):
                fq1, mq1 = model(query1, return_fmap=True)
                fq2, mq2 = model(query2, return_fmap=True)
                fa, ma = model(qaux, return_fmap=True)
                fr1, mr1 = model(reference1, return_fmap=True)
                fr2, mr2 = model(reference2, return_fmap=True)
                m = unwrap_model(model)
                logit_scale = m.logit_scale.exp()

                loss1 = loss_function(fq1, fr1, logit_scale)
                loss2 = loss_function(fq1, fq2, logit_scale)
                loss3 = loss_function(fr1, fr2, logit_scale)
                loss4 = loss_function(fr1, fq2, logit_scale)
                loss_global = loss1 + 0.5 * loss2 + 0.5 * loss3 + 0.25 * loss4

                s180 = local_obj.forward_sim(mq2, mr1, K=cfg.sscl180_K)
                loss_180 = local_obj.ce_from_sim(s180, logit_scale)
                stats180 = local_obj.margin_stats(s180.detach())

                saux = local_obj.forward_sim(ma, mr1, K=cfg.aux_K)
                loss_aux = local_obj.ce_from_sim(saux, logit_scale)
                statsaux = local_obj.margin_stats(saux.detach())

                loss_distill = torch.zeros([], device=cfg.device)
                d_stats = {"res_gain_mean": 0.0, "res_pos_ratio": 0.0, "res_w_mean": 0.0}
                if distill_beta > 0:
                    distill_terms = []
                    d_stats_list = []
                    if cfg.distill_on in ["q2", "both"]:
                        fq2_n = F.normalize(fq2.float(), dim=-1)
                        fr1_n = F.normalize(fr1.float(), dim=-1)
                        dloss, ds = residual_aware_lcr_distill(
                            fq2_n @ fr1_n.T, s180.float(), cfg.distill_lcr_lambda,
                            cfg.tau_teacher, cfg.tau_student, cfg.residual_weight_mode, cfg.residual_soft_tau)
                        distill_terms.append(dloss)
                        d_stats_list.append(ds)
                    if cfg.distill_on in ["aux", "both"]:
                        fa_n = F.normalize(fa.float(), dim=-1)
                        fr1_n = F.normalize(fr1.float(), dim=-1)
                        dloss, ds = residual_aware_lcr_distill(
                            fa_n @ fr1_n.T, saux.float(), cfg.distill_lcr_lambda,
                            cfg.tau_teacher, cfg.tau_student, cfg.residual_weight_mode, cfg.residual_soft_tau)
                        distill_terms.append(dloss)
                        d_stats_list.append(ds)
                    if distill_terms:
                        loss_distill = sum(distill_terms) / len(distill_terms)
                        d_stats = {k: sum(x[k] for x in d_stats_list) / len(d_stats_list) for k in d_stats_list[0]}

                loss = loss_global + sscl180_lambda * loss_180 + aux_lambda * loss_aux + distill_beta * loss_distill
            scaler.scale(loss).backward()
            if cfg.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            raise RuntimeError("This script currently expects mixed precision. Pass --mixed_precision 1.")

        if cfg.scheduler in ["polynomial", "cosine", "constant"] and scheduler is not None:
            scheduler.step()

        bsz = query1.size(0)
        meters["loss"].update(float(loss.detach().cpu()), bsz)
        meters["global"].update(float(loss_global.detach().cpu()), bsz)
        meters["sscl180"].update(float(loss_180.detach().cpu()), bsz)
        meters["aux"].update(float(loss_aux.detach().cpu()), bsz)
        meters["distill"].update(float(loss_distill.detach().cpu()), bsz)
        meters["m180"].update(stats180["margin_mean"], bsz)
        meters["maux"].update(statsaux["margin_mean"], bsz)
        meters["rgain"].update(d_stats["res_gain_mean"], bsz)
        meters["rpos"].update(d_stats["res_pos_ratio"], bsz)
        meters["aux_fov"].update(float(aux_fov[0].item()), bsz)

        if cfg.verbose:
            bar.set_postfix(ordered_dict={
                "loss": f"{meters['loss'].avg:.4f}",
                "G": f"{meters['global'].avg:.4f}",
                "L180": f"{meters['sscl180'].avg:.4f}",
                "Laux": f"{meters['aux'].avg:.4f}",
                "D": f"{meters['distill'].avg:.4f}",
                "m180": f"{meters['m180'].avg:.3f}",
                "maux": f"{meters['maux'].avg:.3f}",
                "rg": f"{meters['rgain'].avg:.3f}",
                "rp": f"{meters['rpos'].avg:.2f}",
                "fov": f"{meters['aux_fov'].avg:.0f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })
    if cfg.verbose:
        bar.close()
    return {k: v.avg for k, v in meters.items()}


def choose_aux_fov(cfg: Configuration, epoch: int) -> float:
    if not cfg.use_aux:
        return cfg.train_fov
    if cfg.aux_schedule == "random":
        return float(random.choice(cfg.aux_fovs))
    return float(cfg.aux_fovs[(epoch - 1) % len(cfg.aux_fovs)])


def make_run_name(cfg: Configuration) -> str:
    if cfg.run_name:
        return cfg.run_name
    name = f"bfov_sscl180K{cfg.sscl180_K}_l{cfg.sscl180_lambda}_auxK{cfg.aux_K}_l{cfg.aux_lambda}"
    name += "_fovs" + "-".join(str(int(x)) if float(x).is_integer() else str(x) for x in cfg.aux_fovs)
    if cfg.use_distill:
        name += f"_resdistill_{cfg.distill_on}_b{cfg.distill_beta}_lcr{cfg.distill_lcr_lambda}_{cfg.residual_weight_mode}"
    return name


def main():
    cfg = parse_args()
    check_required_files(cfg)
    run_name = make_run_name(cfg)
    model_path = os.path.join(cfg.model_path, run_name, cfg.model, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(model_path, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(model_path, "train.py"))
    sys.stdout = Logger(os.path.join(model_path, "log.txt"))
    setup_system(seed=cfg.seed, cudnn_benchmark=cfg.cudnn_benchmark, cudnn_deterministic=cfg.cudnn_deterministic)

    print("========== Balanced-FoV SSCL CVUSA Training ==========")
    print("Model:", cfg.model)
    print("Dataset folder:", cfg.data_folder)
    print("Save folder:", model_path)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count visible:", torch.cuda.device_count())
    print("Configured gpu_ids:", cfg.gpu_ids)
    print("Train FoV:", cfg.train_fov, "Eval FoV:", cfg.eval_fov)
    print("SSCL180: K", cfg.sscl180_K, "lambda", cfg.sscl180_lambda)
    print("Aux: use", cfg.use_aux, "fovs", cfg.aux_fovs, "schedule", cfg.aux_schedule, "K", cfg.aux_K, "lambda", cfg.aux_lambda)
    print("Distill: use", cfg.use_distill, "on", cfg.distill_on, "beta", cfg.distill_beta, "lcr_lambda", cfg.distill_lcr_lambda, "mode", cfg.residual_weight_mode)

    model = TimmModelBalanced(cfg.model, pretrained=False, img_size=cfg.img_size, random_fov=cfg.random_fov)
    if cfg.checkpoint_start is None:
        load_local_timm_pretrain(model, cfg.local_pretrain_path)
    else:
        load_resume_checkpoint(model, cfg.checkpoint_start)

    data_config = model.get_config()
    print("Data config:", data_config)
    mean, std = data_config["mean"], data_config["std"]
    image_size_sat = (cfg.img_size, cfg.img_size)
    new_width = cfg.img_size * 2
    new_height = round((224 / 1232) * new_width)
    img_size_ground = (new_height, new_width)

    if cfg.grad_checkpointing:
        model.set_grad_checkpointing(True)

    available_gpu_count = torch.cuda.device_count()
    valid_gpu_ids = tuple(g for g in cfg.gpu_ids if g < available_gpu_count)
    if available_gpu_count > 1 and len(valid_gpu_ids) > 1:
        print("Use DataParallel on GPUs:", valid_gpu_ids)
        model = torch.nn.DataParallel(model, device_ids=valid_gpu_ids)
    else:
        print("Use single GPU or CPU.")
    model = model.to(cfg.device)

    print("Image Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean:", mean)
    print("Std:", std)

    sat_train1, sat_train2, ground_train1, ground_train2 = get_transforms_train_congeo(
        image_size_sat, img_size_ground, mean=mean, std=std, fov=cfg.train_fov)
    aux_transforms = {}
    for fov in cfg.aux_fovs:
        _, _, _, g_aux = get_transforms_train_congeo(image_size_sat, img_size_ground, mean=mean, std=std, fov=fov)
        aux_transforms[float(fov)] = g_aux

    train_dataset = CVUSADatasetTrainConGeoBalancedAux(
        data_folder=cfg.data_folder,
        transforms_query1=ground_train1,
        transforms_query2=ground_train2,
        transforms_reference1=sat_train1,
        transforms_reference2=sat_train2,
        transforms_query_aux_by_fov=aux_transforms,
        aux_fovs=cfg.aux_fovs,
        prob_flip=cfg.prob_flip,
        prob_rotate=cfg.prob_rotate,
        shuffle_batch_size=cfg.batch_size,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                  shuffle=not cfg.custom_sampling, pin_memory=True, drop_last=True)

    sat_val, ground_val = get_transforms_val(image_size_sat, img_size_ground, mean=mean, std=std, fov=cfg.eval_fov)
    reference_dataset_test = CVUSADatasetEval(cfg.data_folder, split="test", img_type="reference", transforms=sat_val)
    reference_dataloader_test = DataLoader(reference_dataset_test, batch_size=cfg.batch_size_eval, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
    query_dataset_test = CVUSADatasetEval(cfg.data_folder, split="test", img_type="query", transforms=ground_val)
    query_dataloader_test = DataLoader(query_dataset_test, batch_size=cfg.batch_size_eval, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)

    print("Train Images:", len(train_dataset))
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    if cfg.gps_sample:
        print("Load GPS dict:", cfg.gps_dict_path)
        with open(cfg.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    if cfg.sim_sample:
        query_dataset_train = CVUSADatasetEval(cfg.data_folder, split="train", img_type="query", transforms=ground_val)
        query_dataloader_train = DataLoader(query_dataset_train, batch_size=cfg.batch_size_eval, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        reference_dataset_train = CVUSADatasetEval(cfg.data_folder, split="train", img_type="reference", transforms=sat_val)
        reference_dataloader_train = DataLoader(reference_dataset_train, batch_size=cfg.batch_size_eval, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        print("Reference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))
    else:
        reference_dataloader_train = None
        query_dataloader_train = None

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn, device=cfg.device)
    local_obj = StripContrastiveObjective(K=2)
    scaler = GradScaler("cuda", init_scale=2.0 ** 10) if cfg.mixed_precision else None

    if cfg.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=cfg.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train_steps = len(train_dataloader) * cfg.epochs
    warmup_steps = len(train_dataloader) * cfg.warmup_epochs
    if cfg.scheduler == "polynomial":
        print("Scheduler: polynomial - max LR:", cfg.lr, "- end LR:", cfg.lr_end)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_training_steps=train_steps, lr_end=cfg.lr_end, power=1.5, num_warmup_steps=warmup_steps)
    elif cfg.scheduler == "cosine":
        print("Scheduler: cosine - max LR:", cfg.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_steps, num_warmup_steps=warmup_steps)
    elif cfg.scheduler == "constant":
        print("Scheduler: constant - max LR:", cfg.lr)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    print("Warmup Epochs:", cfg.warmup_epochs, "- Warmup Steps:", warmup_steps)
    print("Train Epochs:", cfg.epochs, "- Train Steps:", train_steps)

    if cfg.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict, neighbour_select=cfg.neighbour_select, neighbour_range=cfg.neighbour_range)

    best_score = 0.0
    for epoch in range(1, cfg.epochs + 1):
        aux_fov = choose_aux_fov(cfg, epoch)
        train_dataloader.dataset.set_aux_fov(aux_fov)
        print("\n{}[Epoch: {} | aux_fov: {}]{}".format(30 * "-", epoch, aux_fov, 30 * "-"))
        loss_dict = train_one_epoch(cfg, model, train_dataloader, loss_function, local_obj, optimizer, epoch, scheduler, scaler)
        print(
            "Epoch: {}, Loss={:.4f}, Global={:.4f}, SSCL180={:.4f}, Aux={:.4f}, Distill={:.4f}, "
            "M180={:.4f}, MAux={:.4f}, RGain={:.4f}, RPos={:.4f}, AuxFoV={:.1f}, Lr={:.8f}".format(
                epoch, loss_dict["loss"], loss_dict["global"], loss_dict["sscl180"], loss_dict["aux"],
                loss_dict["distill"], loss_dict["m180"], loss_dict["maux"], loss_dict["rgain"],
                loss_dict["rpos"], loss_dict["aux_fov"], optimizer.param_groups[0]["lr"]
            )
        )

        if cfg.save_latest:
            save_model_state(model, os.path.join(model_path, "weights_latest.pth"))
        if cfg.save_every_epoch:
            save_model_state(model, os.path.join(model_path, f"weights_epoch_{epoch:03d}.pth"))

        if (epoch % cfg.eval_every_n_epoch == 0 and epoch != 0) or epoch == cfg.epochs:
            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))
            r1_test = evaluate(cfg, model, reference_dataloader_test, query_dataloader_test, ranks=[1, 5, 10], step_size=1000, cleanup=True)
            gc.collect(); torch.cuda.empty_cache()
            if cfg.sim_sample:
                r1_train, sim_dict = calc_sim(cfg, model, reference_dataloader_train, query_dataloader_train, ranks=[1, 5, 10], step_size=1000, cleanup=True)
                gc.collect(); torch.cuda.empty_cache()
            if r1_test > best_score:
                best_score = r1_test
                save_model_state(model, os.path.join(model_path, f"weights_best_e{epoch:03d}_{r1_test:.4f}.pth"))
                save_model_state(model, os.path.join(model_path, "weights_best.pth"))
                print("New best R@1:", best_score)

        if cfg.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict, neighbour_select=cfg.neighbour_select, neighbour_range=cfg.neighbour_range)

    save_model_state(model, os.path.join(model_path, "weights_end.pth"))
    print("Training finished. Best R@1:", best_score)
    print("All outputs saved to:", model_path)


if __name__ == "__main__":
    main()
