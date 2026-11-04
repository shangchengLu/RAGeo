#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAGeo training on VIGOR.

Design constraints
------------------
1) Keep the original ConGeo global objective unchanged:
     L(q1,r1) + 0.5 L(q1,q2_180) + 0.5 L(r1,r2) + 0.25 L(r1,q2_180)

2) Keep the RAGeo spatial-strip supervision (SS) unchanged:
     lambda_180 * L_strip(q2_180, r1)

3) Keep the RAGeo FoV-balanced spatial supervision (BF) unchanged:
     q_aux is sampled per epoch from {360,180,90,70}
     lambda_aux * L_strip(q_aux, r1)

4) Keep the RAGeo margin-aware spatial distillation (MSD) unchanged:
     residual-aware local-to-global distillation on the configured branch
     (formal RAGeo setting: distill_on="aux").

5) VIGOR-specific changes are DATA/PROTOCOL adaptations only:
   - same-area / cross-area split
   - VIGOR 384x768 ground preprocessing
   - VIGOR one-to-many evaluation labels
   - path layout under VIGOR_verified/{city}/...
   No new scientific loss or RAGeo component is introduced.

The script also supports a bounded smoke run through --max_steps_per_epoch and
--skip_eval, so the exact same training implementation is used for smoke tests.
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
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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

from rageo.evaluate.vigor import calc_sim, evaluate
from rageo.loss import InfoNCE
from rageo.model import TimmModel_ConGeo
from rageo.transforms import get_transforms_train_congeo, get_transforms_val
from rageo.utils import Logger, setup_system


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class Configuration:
    dataset: str = "vigor"
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    local_pretrain_path: str = (
        "./pretrained/pytorch_model.bin"
    )
    img_size: int = 384

    seed: int = 42
    epochs: int = 60
    batch_size: int = 16
    batch_size_eval: int = 16
    num_workers: int = 4
    gpu_ids: Tuple[int, ...] = (0,)
    mixed_precision: bool = True
    verbose: bool = True

    # VIGOR protocol
    same_area: bool = True

    # Similarity / GPS sampling: keep ConGeo behavior for formal training.
    custom_sampling: bool = True
    gps_sample: bool = True
    sim_sample: bool = True
    neighbour_select: int = 64
    neighbour_range: int = 128
    gps_dict_path: str = ""

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

    # IMPORTANT:
    # For formal jobs this may point to node-local staged VIGOR_verified.
    data_folder: str = "./datasets/VIGOR"
    prob_rotate: float = 0.75
    prob_flip: float = 0.5
    model_path: str = "./balanced_fov_sscl_vigor"
    checkpoint_start: Optional[str] = None
    save_every_epoch: bool = True
    save_latest: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # ConGeo / RAGeo FoV setup.
    train_fov: float = 180.0
    eval_fov: float = 90.0
    random_fov: bool = False

    # RAGeo SS.
    sscl180_K: int = 2
    sscl180_lambda: float = 0.25

    # RAGeo BF.
    use_aux: bool = True
    aux_fovs: Tuple[float, ...] = (360.0, 180.0, 90.0, 70.0)
    aux_schedule: str = "cycle"
    aux_K: int = 2
    aux_lambda: float = 0.05
    local_warmup_epochs: int = 5

    # RAGeo MSD.
    use_distill: bool = True
    distill_on: str = "aux"
    distill_beta: float = 0.01
    distill_warmup_epochs: int = 5
    distill_lcr_lambda: float = 3.0
    tau_teacher: float = 1.0
    tau_student: float = 1.0
    residual_weight_mode: str = "positive"
    residual_soft_tau: float = 0.2

    run_name: str = ""

    # Engineering-only controls for smoke testing.
    max_steps_per_epoch: int = 0   # 0 = full epoch
    skip_eval: bool = False


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
    parser.add_argument("--same_area", type=str, default="1")
    parser.add_argument("--gps_dict_path", type=str, default="")
    parser.add_argument("--local_pretrain_path", type=str, default=Configuration.local_pretrain_path)
    parser.add_argument("--checkpoint_start", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=Configuration.model_path)
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--epochs", type=int, default=Configuration.epochs)
    parser.add_argument("--batch_size", type=int, default=Configuration.batch_size)
    parser.add_argument("--batch_size_eval", type=int, default=Configuration.batch_size_eval)
    parser.add_argument("--num_workers", type=int, default=Configuration.num_workers)
    parser.add_argument("--gpu_ids", type=str, default="0")

    parser.add_argument("--lr", type=float, default=Configuration.lr)
    parser.add_argument(
        "--scheduler",
        type=str,
        default=Configuration.scheduler,
        choices=["cosine", "polynomial", "constant", "none"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=Configuration.warmup_epochs)
    parser.add_argument("--eval_every_n_epoch", type=int, default=Configuration.eval_every_n_epoch)

    parser.add_argument("--train_fov", type=float, default=Configuration.train_fov)
    parser.add_argument("--eval_fov", type=float, default=Configuration.eval_fov)

    parser.add_argument("--sscl180_K", type=int, default=Configuration.sscl180_K)
    parser.add_argument("--sscl180_lambda", type=float, default=Configuration.sscl180_lambda)

    parser.add_argument("--use_aux", type=str, default="1")
    parser.add_argument("--aux_fovs", type=str, default="360 180 90 70")
    parser.add_argument(
        "--aux_schedule",
        type=str,
        default=Configuration.aux_schedule,
        choices=["cycle", "random"],
    )
    parser.add_argument("--aux_K", type=int, default=Configuration.aux_K)
    parser.add_argument("--aux_lambda", type=float, default=Configuration.aux_lambda)
    parser.add_argument("--local_warmup_epochs", type=int, default=Configuration.local_warmup_epochs)

    parser.add_argument("--use_distill", type=str, default="1")
    parser.add_argument(
        "--distill_on",
        type=str,
        default=Configuration.distill_on,
        choices=["aux", "q2", "both"],
    )
    parser.add_argument("--distill_beta", type=float, default=Configuration.distill_beta)
    parser.add_argument("--distill_warmup_epochs", type=int, default=Configuration.distill_warmup_epochs)
    parser.add_argument("--distill_lcr_lambda", type=float, default=Configuration.distill_lcr_lambda)
    parser.add_argument("--tau_teacher", type=float, default=Configuration.tau_teacher)
    parser.add_argument("--tau_student", type=float, default=Configuration.tau_student)
    parser.add_argument(
        "--residual_weight_mode",
        type=str,
        default=Configuration.residual_weight_mode,
        choices=["positive", "soft", "all"],
    )
    parser.add_argument("--residual_soft_tau", type=float, default=Configuration.residual_soft_tau)

    parser.add_argument("--mixed_precision", type=str, default="1")
    parser.add_argument("--custom_sampling", type=str, default="1")
    parser.add_argument("--gps_sample", type=str, default="1")
    parser.add_argument("--sim_sample", type=str, default="1")
    parser.add_argument("--save_every_epoch", type=str, default="1")
    parser.add_argument("--save_latest", type=str, default="1")
    parser.add_argument("--verbose", type=str, default="1")

    # Engineering-only smoke controls.
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--skip_eval", type=str, default="0")

    args = parser.parse_args()

    cfg = Configuration()
    cfg.data_folder = args.data_folder
    cfg.same_area = str2bool_int(args.same_area)
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

    cfg.max_steps_per_epoch = args.max_steps_per_epoch
    cfg.skip_eval = str2bool_int(args.skip_eval)

    if not cfg.gps_dict_path:
        name = "gps_dict_same.pkl" if cfg.same_area else "gps_dict_cross.pkl"
        cfg.gps_dict_path = os.path.join(cfg.data_folder, name)

    return cfg


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

_BAD_IMAGE_LOG = set()


def _warn_bad_image(path, context=""):
    key = (context, str(path))
    if key not in _BAD_IMAGE_LOG:
        _BAD_IMAGE_LOG.add(key)
        print(f"[BAD_IMAGE] {context}: {path}", flush=True)


def _imread_rgb_or_none(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _fallback_eval_image(img_type):
    if img_type == "query":
        return np.zeros((1024, 2048, 3), dtype=np.uint8)
    return np.zeros((640, 640, 3), dtype=np.uint8)


def check_required_files(cfg: Configuration):
    if not os.path.isdir(cfg.data_folder):
        raise FileNotFoundError(f"Dataset root not found: {cfg.data_folder}")

    cities = ["Chicago", "NewYork", "SanFrancisco", "Seattle"]
    for city in cities:
        city_dir = os.path.join(cfg.data_folder, city)
        split_dir = os.path.join(cfg.data_folder, "splits", city)
        if not os.path.isdir(city_dir):
            raise FileNotFoundError(f"City directory not found: {city_dir}")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

    weight_path = cfg.checkpoint_start if cfg.checkpoint_start else cfg.local_pretrain_path
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Model weight not found: {weight_path}")

    if cfg.gps_sample and not os.path.isfile(cfg.gps_dict_path):
        raise FileNotFoundError(
            f"GPS dictionary not found: {cfg.gps_dict_path}\n"
            f"Pass --gps_dict_path explicitly, or use --gps_sample 0 for smoke testing."
        )

    print("Required paths checked.")


# -----------------------------------------------------------------------------
# VIGOR dataset: original ConGeo protocol + one RAGeo q_aux branch
# -----------------------------------------------------------------------------

class VigorDatasetTrainConGeoBalancedAux(Dataset):
    """
    VIGOR ConGeo training dataset with one additional RAGeo q_aux branch.

    Scientific semantics are unchanged:
      q1, q2, r1, r2 follow the original ConGeo VIGOR pipeline.
      q_aux is an additional transformed view of THE SAME already-loaded ground image.

    No additional disk read is introduced for q_aux.
    """

    def __init__(
        self,
        data_folder,
        same_area=True,
        transforms_query1=None,
        transforms_query2=None,
        transforms_reference1=None,
        transforms_reference2=None,
        transforms_query_aux_by_fov=None,
        aux_fovs=(360.0, 180.0, 90.0, 70.0),
        prob_flip=0.0,
        prob_rotate=0.0,
        shuffle_batch_size=128,
    ):
        super().__init__()

        self.data_folder = data_folder
        self.same_area = same_area
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query1 = transforms_query1
        self.transforms_query2 = transforms_query2
        self.transforms_reference1 = transforms_reference1
        self.transforms_reference2 = transforms_reference2
        self.transforms_query_aux_by_fov = transforms_query_aux_by_fov or {}
        self.aux_fovs = tuple(float(x) for x in aux_fovs)
        self.current_aux_fov = self.aux_fovs[0]

        if same_area:
            self.cities = ["Chicago", "NewYork", "SanFrancisco", "Seattle"]
        else:
            self.cities = ["NewYork", "Seattle"]

        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(
                f"{data_folder}/splits/{city}/satellite_list.txt",
                header=None,
                sep=r"\s+",
            )
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(
                lambda x: f"{data_folder}/{city}/satellite/{x.sat}",
                axis=1,
            )
            sat_list.append(df_tmp)

        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))

        ground_list = []
        for city in self.cities:
            if same_area:
                split_file = f"{data_folder}/splits/{city}/same_area_balanced_train.txt"
            else:
                split_file = f"{data_folder}/splits/{city}/pano_label_balanced.txt"

            df_tmp = pd.read_csv(split_file, header=None, sep=r"\s+")
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(
                columns={
                    0: "ground",
                    1: "sat",
                    4: "sat_np1",
                    7: "sat_np2",
                    10: "sat_np3",
                }
            )

            df_tmp["path_ground"] = df_tmp.apply(
                lambda x: f"{data_folder}/{city}/panorama/{x.ground}",
                axis=1,
            )
            df_tmp["path_sat"] = df_tmp.apply(
                lambda x: f"{data_folder}/{city}/satellite/{x.sat}",
                axis=1,
            )

            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[sat_n] = df_tmp[sat_n].map(sat2idx)

            ground_list.append(df_tmp)

        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)

        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))

        # Preserve original ConGeo VIGOR primary-positive training pairing.
        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)
        for pair in self.pairs:
            self.idx2pairs[pair[1]].append(pair)

        # Retained for protocol completeness / diagnostics.
        self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values
        self.samples = copy.deepcopy(self.pairs)

    def set_aux_fov(self, fov: float):
        fov = float(fov)
        if fov not in self.transforms_query_aux_by_fov:
            raise ValueError(
                f"Aux FoV {fov} not configured. "
                f"Available: {list(self.transforms_query_aux_by_fov.keys())}"
            )
        self.current_aux_fov = fov

    def __getitem__(self, index):
        # VIGOR_verified should contain no missing referenced images.
        # Retry is retained only as a defensive engineering fallback.
        for _ in range(10):
            idx_ground, idx_sat = self.samples[index]

            query_path = self.idx2ground_path[idx_ground]
            reference_path = self.idx2sat_path[idx_sat]

            # IMPORTANT for shared filesystems:
            # each physical image is read exactly ONCE per sample.
            query_img = _imread_rgb_or_none(query_path)
            reference_img = _imread_rgb_or_none(reference_path)

            if query_img is not None and reference_img is not None:
                break

            if query_img is None:
                _warn_bad_image(query_path, "rageo-vigor-train-query")
            if reference_img is None:
                _warn_bad_image(reference_path, "rageo-vigor-train-reference")

            index = random.randint(0, len(self.samples) - 1)
        else:
            raise RuntimeError("Too many bad VIGOR training images were sampled consecutively.")

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # Multiple views are generated from the same in-memory image.
        query_img1 = self.transforms_query1(image=query_img)["image"]
        query_img2 = self.transforms_query2(image=query_img)["image"]
        query_aux = self.transforms_query_aux_by_fov[self.current_aux_fov](image=query_img)["image"]
        reference_img1 = self.transforms_reference1(image=reference_img)["image"]
        reference_img2 = self.transforms_reference2(image=reference_img)["image"]

        # Preserve original ConGeo augmentation behavior.
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])

            reference_img1 = torch.rot90(reference_img1, k=r, dims=(1, 2))

            _, _, w1 = query_img1.shape
            query_img1 = torch.roll(query_img1, shifts=-(w1 // 4) * r, dims=2)

            _, _, w2 = query_img2.shape
            query_img2 = torch.roll(query_img2, shifts=-(w2 // 4) * r, dims=2)

            _, _, wa = query_aux.shape
            query_aux = torch.roll(query_aux, shifts=-(wa // 4) * r, dims=2)

        label = torch.tensor(idx_sat, dtype=torch.long)
        aux_fov_tensor = torch.tensor(self.current_aux_fov, dtype=torch.float32)

        return (
            query_img1,
            query_img2,
            query_aux,
            reference_img1,
            reference_img2,
            label,
            aux_fov_tensor,
        )

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        """
        Preserve the original ConGeo VIGOR class-unique batch sampler.
        This keeps each satellite class unique inside a mini-batch, which preserves
        the diagonal-positive assumption used by InfoNCE, SS and MSD.
        """

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)
        idx2pair_pool = copy.deepcopy(self.idx2pairs)
        neighbour_split = neighbour_select // 2

        similarity_pool = copy.deepcopy(sim_dict) if sim_dict is not None else None

        random.shuffle(pair_pool)

        pairs_epoch = set()
        idx_batch = set()
        batches = []
        current_batch = []

        break_counter = 0
        pbar = tqdm()

        while True:
            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                _, idx = pair

                if (
                    idx not in idx_batch
                    and pair not in pairs_epoch
                    and len(current_batch) < self.shuffle_batch_size
                ):
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    if idx in idx2pair_pool and pair in idx2pair_pool[idx]:
                        idx2pair_pool[idx].remove(pair)

                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                        near_always = copy.deepcopy(near_similarity[:neighbour_split])
                        near_random = copy.deepcopy(near_similarity[neighbour_split:])
                        random.shuffle(near_random)
                        near_random = near_random[:neighbour_split]
                        near_similarity_select = near_always + near_random

                        for idx_near in near_similarity_select:
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            if idx_near not in idx_batch:
                                near_pairs = copy.deepcopy(idx2pair_pool.get(idx_near, []))
                                random.shuffle(near_pairs)

                                for near_pair in near_pairs:
                                    idx_batch.add(idx_near)
                                    current_batch.append(near_pair)
                                    pairs_epoch.add(near_pair)

                                    if near_pair in idx2pair_pool[idx_near]:
                                        idx2pair_pool[idx_near].remove(near_pair)

                                    if idx_near in similarity_pool[idx]:
                                        similarity_pool[idx].remove(idx_near)

                                    # Preserve original behavior: only one view for this satellite.
                                    break

                    break_counter = 0

                else:
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
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

        print("pair_pool:", len(pair_pool))
        print(
            "Original Length: {} - Length after Shuffle: {}".format(
                len(self.pairs), len(self.samples)
            )
        )
        print("Break Counter:", break_counter)
        print(
            "Pairs left out of last batch to avoid creating noise:",
            len(self.pairs) - len(self.samples),
        )
        if self.samples:
            print(
                "First Element ID: {} - Last Element ID: {}".format(
                    self.samples[0][1], self.samples[-1][1]
                )
            )


class VigorDatasetEvalVerified(Dataset):
    """
    VIGOR evaluator with the same labels/protocol as ConGeo, but with paths matching
    the verified dataset layout:

      {data_folder}/{city}/panorama/*
      {data_folder}/{city}/satellite/*

    Evaluation remains one-to-many through four GT satellite ids per ground query.
    """

    def __init__(self, data_folder, split, img_type, same_area=True, transforms=None):
        super().__init__()

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if img_type not in {"query", "reference"}:
            raise ValueError("img_type must be 'query' or 'reference'")

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        if same_area:
            self.cities = ["Chicago", "NewYork", "SanFrancisco", "Seattle"]
        else:
            if split == "train":
                self.cities = ["NewYork", "Seattle"]
            else:
                self.cities = ["Chicago", "SanFrancisco"]

        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(
                f"{data_folder}/splits/{city}/satellite_list.txt",
                header=None,
                sep=r"\s+",
            )
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(
                lambda x: f"{data_folder}/{city}/satellite/{x.sat}",
                axis=1,
            )
            sat_list.append(df_tmp)

        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))

        ground_list = []
        for city in self.cities:
            if same_area:
                split_file = f"{data_folder}/splits/{city}/same_area_balanced_{split}.txt"
            else:
                split_file = f"{data_folder}/splits/{city}/pano_label_balanced.txt"

            df_tmp = pd.read_csv(split_file, header=None, sep=r"\s+")
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(
                columns={
                    0: "ground",
                    1: "sat",
                    4: "sat_np1",
                    7: "sat_np2",
                    10: "sat_np3",
                }
            )

            df_tmp["path_ground"] = df_tmp.apply(
                lambda x: f"{data_folder}/{city}/panorama/{x.ground}",
                axis=1,
            )

            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[sat_n] = df_tmp[sat_n].map(sat2idx)

            ground_list.append(df_tmp)

        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)

        if self.img_type == "reference":
            if split == "train":
                # Preserve ConGeo sim-sampling behavior: references actually used for training.
                self.label = self.df_ground["sat"].unique()
                self.images = [self.idx2sat_path[idx] for idx in self.label]
            else:
                # All satellite images in the test cities.
                self.images = self.df_sat["path"].values
                self.label = self.df_sat.index.values

        else:  # query
            self.images = self.df_ground["path_ground"].values
            self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.label[index]

        img = _imread_rgb_or_none(img_path)
        if img is None:
            _warn_bad_image(img_path, f"eval-{self.split}-{self.img_type}")
            img = _fallback_eval_image(self.img_type)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.images)


# -----------------------------------------------------------------------------
# RAGeo model / objectives: copied conceptually unchanged from formal CVUSA code
# -----------------------------------------------------------------------------

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
        sim = torch.einsum("bkc,jmc->bjkm", q_strips, r_strips)
        sim_max = sim.max(dim=3).values
        return sim_max.mean(dim=2)

    def forward_sim(
        self,
        q_fmap: torch.Tensor,
        r_fmap: torch.Tensor,
        K: Optional[int] = None,
    ) -> torch.Tensor:
        K = self.K if K is None else K
        q_strips = self.strip_pool(q_fmap, K)
        r_strips = self.strip_pool(r_fmap, K)
        return self.compute_local_sim_matrix(q_strips, r_strips)

    @staticmethod
    def ce_from_sim(sim_matrix: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        B = sim_matrix.size(0)
        labels = torch.arange(B, dtype=torch.long, device=sim_matrix.device)
        logits = sim_matrix * logit_scale
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

    @staticmethod
    def margin_stats(sim_matrix: torch.Tensor) -> Dict[str, float]:
        B = sim_matrix.size(0)
        labels = torch.arange(B, device=sim_matrix.device)

        pos = sim_matrix[labels, labels]
        neg = sim_matrix.masked_fill(
            torch.eye(B, device=sim_matrix.device, dtype=torch.bool),
            -1e4,
        ).max(dim=1).values

        margin = pos - neg

        return {
            "margin_mean": float(margin.mean().detach().cpu()),
            "margin_pos_ratio": float((margin > 0).float().mean().detach().cpu()),
        }


def row_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def residual_aware_lcr_distill(
    s_global: torch.Tensor,
    s_local: torch.Tensor,
    lcr_lambda: float = 3.0,
    tau_teacher: float = 1.0,
    tau_student: float = 1.0,
    weight_mode: str = "positive",
    soft_tau: float = 0.2,
):
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
                weights = torch.zeros_like(residual_gain)

        elif weight_mode == "soft":
            weights = torch.sigmoid(
                residual_gain / max(soft_tau, 1e-6)
            )

        else:
            raise ValueError(f"Unknown residual weight mode: {weight_mode}")

        p_teacher = F.softmax(teacher_logits / tau_teacher, dim=1)

    log_p_student = F.log_softmax(student_logits / tau_student, dim=1)

    per_row_kl = F.kl_div(
        log_p_student,
        p_teacher,
        reduction="none",
    ).sum(dim=1)

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


def train_one_epoch(
    cfg: Configuration,
    model,
    dataloader,
    loss_function,
    local_obj,
    optimizer,
    epoch: int,
    scheduler=None,
    scaler=None,
):
    model.train()

    meters = {
        k: AverageMeter()
        for k in [
            "loss",
            "global",
            "sscl180",
            "aux",
            "distill",
            "m180",
            "maux",
            "rgain",
            "rpos",
            "aux_fov",
        ]
    }

    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(dataloader, total=len(dataloader)) if cfg.verbose else dataloader

    sscl180_lambda = (
        cfg.sscl180_lambda
        if epoch > cfg.local_warmup_epochs
        else 0.0
    )

    aux_lambda = (
        cfg.aux_lambda
        if (cfg.use_aux and epoch > cfg.local_warmup_epochs)
        else 0.0
    )

    distill_beta = (
        cfg.distill_beta
        if (cfg.use_distill and epoch > cfg.distill_warmup_epochs)
        else 0.0
    )

    processed_steps = 0

    for (
        query1,
        query2,
        qaux,
        reference1,
        reference2,
        ids,
        aux_fov,
    ) in bar:

        query1 = query1.to(cfg.device, non_blocking=True)
        query2 = query2.to(cfg.device, non_blocking=True)
        qaux = qaux.to(cfg.device, non_blocking=True)
        reference1 = reference1.to(cfg.device, non_blocking=True)
        reference2 = reference2.to(cfg.device, non_blocking=True)

        if scaler is None:
            raise RuntimeError(
                "This script currently expects mixed precision. "
                "Pass --mixed_precision 1."
            )

        with autocast("cuda"):
            fq1, mq1 = model(query1, return_fmap=True)
            fq2, mq2 = model(query2, return_fmap=True)
            fa, ma = model(qaux, return_fmap=True)
            fr1, mr1 = model(reference1, return_fmap=True)
            fr2, mr2 = model(reference2, return_fmap=True)

            m = unwrap_model(model)
            logit_scale = m.logit_scale.exp()

            # -------------------------------------------------------------
            # ORIGINAL ConGeo global objective -- unchanged.
            # -------------------------------------------------------------
            loss1 = loss_function(fq1, fr1, logit_scale)
            loss2 = loss_function(fq1, fq2, logit_scale)
            loss3 = loss_function(fr1, fr2, logit_scale)
            loss4 = loss_function(fr1, fq2, logit_scale)

            loss_global = (
                loss1
                + 0.5 * loss2
                + 0.5 * loss3
                + 0.25 * loss4
            )

            # -------------------------------------------------------------
            # RAGeo SS -- unchanged.
            # -------------------------------------------------------------
            s180 = local_obj.forward_sim(
                mq2,
                mr1,
                K=cfg.sscl180_K,
            )
            loss_180 = local_obj.ce_from_sim(
                s180,
                logit_scale,
            )
            stats180 = local_obj.margin_stats(s180.detach())

            # -------------------------------------------------------------
            # RAGeo BF -- unchanged.
            # -------------------------------------------------------------
            saux = local_obj.forward_sim(
                ma,
                mr1,
                K=cfg.aux_K,
            )
            loss_aux = local_obj.ce_from_sim(
                saux,
                logit_scale,
            )
            statsaux = local_obj.margin_stats(saux.detach())

            # -------------------------------------------------------------
            # RAGeo MSD -- unchanged.
            # -------------------------------------------------------------
            loss_distill = torch.zeros([], device=cfg.device)

            d_stats = {
                "res_gain_mean": 0.0,
                "res_pos_ratio": 0.0,
                "res_w_mean": 0.0,
            }

            if distill_beta > 0:
                distill_terms = []
                d_stats_list = []

                if cfg.distill_on in ["q2", "both"]:
                    fq2_n = F.normalize(fq2.float(), dim=-1)
                    fr1_n = F.normalize(fr1.float(), dim=-1)

                    dloss, ds = residual_aware_lcr_distill(
                        fq2_n @ fr1_n.T,
                        s180.float(),
                        cfg.distill_lcr_lambda,
                        cfg.tau_teacher,
                        cfg.tau_student,
                        cfg.residual_weight_mode,
                        cfg.residual_soft_tau,
                    )

                    distill_terms.append(dloss)
                    d_stats_list.append(ds)

                if cfg.distill_on in ["aux", "both"]:
                    fa_n = F.normalize(fa.float(), dim=-1)
                    fr1_n = F.normalize(fr1.float(), dim=-1)

                    dloss, ds = residual_aware_lcr_distill(
                        fa_n @ fr1_n.T,
                        saux.float(),
                        cfg.distill_lcr_lambda,
                        cfg.tau_teacher,
                        cfg.tau_student,
                        cfg.residual_weight_mode,
                        cfg.residual_soft_tau,
                    )

                    distill_terms.append(dloss)
                    d_stats_list.append(ds)

                if distill_terms:
                    loss_distill = sum(distill_terms) / len(distill_terms)

                    d_stats = {
                        k: sum(x[k] for x in d_stats_list) / len(d_stats_list)
                        for k in d_stats_list[0]
                    }

            loss = (
                loss_global
                + sscl180_lambda * loss_180
                + aux_lambda * loss_aux
                + distill_beta * loss_distill
            )

        scaler.scale(loss).backward()

        if cfg.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                cfg.clip_grad,
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if (
            cfg.scheduler in ["polynomial", "cosine", "constant"]
            and scheduler is not None
        ):
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

        processed_steps += 1

        if cfg.verbose:
            bar.set_postfix(
                ordered_dict={
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
                }
            )

        if cfg.max_steps_per_epoch > 0 and processed_steps >= cfg.max_steps_per_epoch:
            print(
                f"[ENGINEERING] Stop epoch after "
                f"{cfg.max_steps_per_epoch} step(s)."
            )
            break

    if cfg.verbose:
        bar.close()

    out = {k: v.avg for k, v in meters.items()}
    out["steps"] = processed_steps
    return out


def choose_aux_fov(cfg: Configuration, epoch: int) -> float:
    if not cfg.use_aux:
        return cfg.train_fov

    if cfg.aux_schedule == "random":
        return float(random.choice(cfg.aux_fovs))

    return float(
        cfg.aux_fovs[(epoch - 1) % len(cfg.aux_fovs)]
    )


def make_run_name(cfg: Configuration) -> str:
    if cfg.run_name:
        return cfg.run_name

    protocol = "same" if cfg.same_area else "cross"

    name = (
        f"{protocol}_"
        f"bfov_sscl180K{cfg.sscl180_K}_l{cfg.sscl180_lambda}_"
        f"auxK{cfg.aux_K}_l{cfg.aux_lambda}"
    )

    name += "_fovs" + "-".join(
        str(int(x)) if float(x).is_integer() else str(x)
        for x in cfg.aux_fovs
    )

    if cfg.use_distill:
        name += (
            f"_resdistill_{cfg.distill_on}"
            f"_b{cfg.distill_beta}"
            f"_lcr{cfg.distill_lcr_lambda}"
            f"_{cfg.residual_weight_mode}"
        )

    return name


def main():
    cfg = parse_args()
    check_required_files(cfg)

    run_name = make_run_name(cfg)

    model_path = os.path.join(
        cfg.model_path,
        run_name,
        cfg.model,
        time.strftime("%Y%m%d_%H%M%S"),
    )

    os.makedirs(model_path, exist_ok=True)

    shutil.copyfile(
        os.path.abspath(__file__),
        os.path.join(model_path, "train.py"),
    )

    sys.stdout = Logger(
        os.path.join(model_path, "log.txt")
    )

    setup_system(
        seed=cfg.seed,
        cudnn_benchmark=cfg.cudnn_benchmark,
        cudnn_deterministic=cfg.cudnn_deterministic,
    )

    protocol_name = "Same-Area" if cfg.same_area else "Cross-Area"

    print("========== RAGeo VIGOR Training ==========")
    print("Protocol:", protocol_name)
    print("Model:", cfg.model)
    print("Dataset folder:", cfg.data_folder)
    print("Save folder:", model_path)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count visible:", torch.cuda.device_count())
    print("Configured gpu_ids:", cfg.gpu_ids)
    print("Train FoV:", cfg.train_fov)
    print("Eval FoV:", cfg.eval_fov)

    print(
        "SS:",
        "K", cfg.sscl180_K,
        "lambda", cfg.sscl180_lambda,
    )

    print(
        "BF:",
        "use", cfg.use_aux,
        "fovs", cfg.aux_fovs,
        "schedule", cfg.aux_schedule,
        "K", cfg.aux_K,
        "lambda", cfg.aux_lambda,
    )

    print(
        "MSD:",
        "use", cfg.use_distill,
        "on", cfg.distill_on,
        "beta", cfg.distill_beta,
        "lcr_lambda", cfg.distill_lcr_lambda,
        "tau_t", cfg.tau_teacher,
        "tau_s", cfg.tau_student,
        "mode", cfg.residual_weight_mode,
    )

    print(
        "Engineering:",
        "num_workers", cfg.num_workers,
        "max_steps_per_epoch", cfg.max_steps_per_epoch,
        "skip_eval", cfg.skip_eval,
    )

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------

    model = TimmModelBalanced(
        cfg.model,
        pretrained=False,
        img_size=cfg.img_size,
        random_fov=cfg.random_fov,
    )

    if cfg.checkpoint_start is None:
        load_local_timm_pretrain(
            model,
            cfg.local_pretrain_path,
        )
    else:
        load_resume_checkpoint(
            model,
            cfg.checkpoint_start,
        )

    data_config = model.get_config()

    print("Data config:", data_config)

    mean = data_config["mean"]
    std = data_config["std"]

    # VIGOR preprocessing from ConGeo:
    # satellite 384x384, ground 384x768.
    image_size_sat = (
        cfg.img_size,
        cfg.img_size,
    )

    new_width = cfg.img_size * 2
    new_height = int(
        (1024 / 2048) * new_width
    )

    img_size_ground = (
        new_height,
        new_width,
    )

    if cfg.grad_checkpointing:
        model.set_grad_checkpointing(True)

    available_gpu_count = torch.cuda.device_count()
    valid_gpu_ids = tuple(
        g for g in cfg.gpu_ids
        if g < available_gpu_count
    )

    if available_gpu_count > 1 and len(valid_gpu_ids) > 1:
        print("Use DataParallel on GPUs:", valid_gpu_ids)
        model = torch.nn.DataParallel(
            model,
            device_ids=valid_gpu_ids,
        )
    else:
        print("Use single GPU or CPU.")

    model = model.to(cfg.device)

    print("Image Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean:", mean)
    print("Std:", std)

    # ---------------------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------------------

    (
        sat_train1,
        sat_train2,
        ground_train1,
        ground_train2,
    ) = get_transforms_train_congeo(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std,
        fov=cfg.train_fov,
    )

    aux_transforms = {}

    for fov in cfg.aux_fovs:
        _, _, _, g_aux = get_transforms_train_congeo(
            image_size_sat,
            img_size_ground,
            mean=mean,
            std=std,
            fov=fov,
        )

        aux_transforms[float(fov)] = g_aux

    # ---------------------------------------------------------------------
    # Train dataset
    # ---------------------------------------------------------------------

    train_dataset = VigorDatasetTrainConGeoBalancedAux(
        data_folder=cfg.data_folder,
        same_area=cfg.same_area,
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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=not cfg.custom_sampling,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ---------------------------------------------------------------------
    # Eval dataset
    # ---------------------------------------------------------------------

    # Test transform: explicitly apply the requested evaluation FoV.
    sat_val, ground_val = get_transforms_val(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std,
        fov=cfg.eval_fov,
    )

    # Similarity-sampling transform: preserve the original ConGeo VIGOR
    # training logic, whose train-set similarity refresh uses validation
    # transforms with the default/full FoV rather than the FoV-specific
    # test transform above.
    sat_sim, ground_sim = get_transforms_val(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std,
        fov=0.0,
    )

    reference_dataset_test = VigorDatasetEvalVerified(
        cfg.data_folder,
        split="test",
        img_type="reference",
        same_area=cfg.same_area,
        transforms=sat_val,
    )

    reference_dataloader_test = DataLoader(
        reference_dataset_test,
        batch_size=cfg.batch_size_eval,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    query_dataset_test = VigorDatasetEvalVerified(
        cfg.data_folder,
        split="test",
        img_type="query",
        same_area=cfg.same_area,
        transforms=ground_val,
    )

    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=cfg.batch_size_eval,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    print("Train samples:", len(train_dataset))
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    # ---------------------------------------------------------------------
    # GPS / similarity sampling
    # ---------------------------------------------------------------------

    if cfg.gps_sample:
        print("Load GPS dict:", cfg.gps_dict_path)
        with open(cfg.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    if cfg.sim_sample:
        query_dataset_train = VigorDatasetEvalVerified(
            cfg.data_folder,
            split="train",
            img_type="query",
            same_area=cfg.same_area,
            transforms=ground_sim,
        )

        query_dataloader_train = DataLoader(
            query_dataset_train,
            batch_size=cfg.batch_size_eval,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
        )

        reference_dataset_train = VigorDatasetEvalVerified(
            cfg.data_folder,
            split="train",
            img_type="reference",
            same_area=cfg.same_area,
            transforms=sat_sim,
        )

        reference_dataloader_train = DataLoader(
            reference_dataset_train,
            batch_size=cfg.batch_size_eval,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
        )

        print("Reference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))

    else:
        reference_dataloader_train = None
        query_dataloader_train = None

    # ---------------------------------------------------------------------
    # Loss / optimizer
    # ---------------------------------------------------------------------

    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=cfg.label_smoothing
    )

    loss_function = InfoNCE(
        loss_function=loss_fn,
        device=cfg.device,
    )

    local_obj = StripContrastiveObjective(K=2)

    scaler = (
        GradScaler("cuda", init_scale=2.0 ** 10)
        if cfg.mixed_precision
        else None
    )

    if cfg.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]

        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=cfg.lr,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
        )

    # For bounded smoke tests, scheduler steps should match the number of
    # actually executed batches, not the full VIGOR epoch.
    effective_steps_per_epoch = len(train_dataloader)

    if cfg.max_steps_per_epoch > 0:
        effective_steps_per_epoch = min(
            effective_steps_per_epoch,
            cfg.max_steps_per_epoch,
        )

    train_steps = effective_steps_per_epoch * cfg.epochs
    warmup_steps = effective_steps_per_epoch * cfg.warmup_epochs

    if cfg.scheduler == "polynomial":
        print(
            "Scheduler: polynomial - max LR:",
            cfg.lr,
            "- end LR:",
            cfg.lr_end,
        )

        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=cfg.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps,
        )

    elif cfg.scheduler == "cosine":
        print("Scheduler: cosine - max LR:", cfg.lr)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps,
        )

    elif cfg.scheduler == "constant":
        print("Scheduler: constant - max LR:", cfg.lr)

        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )

    else:
        scheduler = None

    print(
        "Warmup Epochs:",
        cfg.warmup_epochs,
        "- Warmup Steps:",
        warmup_steps,
    )

    print(
        "Train Epochs:",
        cfg.epochs,
        "- Effective Train Steps:",
        train_steps,
    )

    # ---------------------------------------------------------------------
    # Initial shuffle
    # ---------------------------------------------------------------------

    if cfg.custom_sampling:
        train_dataloader.dataset.shuffle(
            sim_dict,
            neighbour_select=cfg.neighbour_select,
            neighbour_range=cfg.neighbour_range,
        )

    # ---------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------

    best_score = 0.0

    for epoch in range(1, cfg.epochs + 1):
        aux_fov = choose_aux_fov(cfg, epoch)
        train_dataloader.dataset.set_aux_fov(aux_fov)

        print(
            "\n{}[Epoch: {} | aux_fov: {}]{}".format(
                30 * "-",
                epoch,
                aux_fov,
                30 * "-",
            )
        )

        loss_dict = train_one_epoch(
            cfg,
            model,
            train_dataloader,
            loss_function,
            local_obj,
            optimizer,
            epoch,
            scheduler,
            scaler,
        )

        print(
            "Epoch: {}, Steps={}, Loss={:.4f}, Global={:.4f}, "
            "SSCL180={:.4f}, Aux={:.4f}, Distill={:.4f}, "
            "M180={:.4f}, MAux={:.4f}, RGain={:.4f}, "
            "RPos={:.4f}, AuxFoV={:.1f}, Lr={:.8f}".format(
                epoch,
                loss_dict["steps"],
                loss_dict["loss"],
                loss_dict["global"],
                loss_dict["sscl180"],
                loss_dict["aux"],
                loss_dict["distill"],
                loss_dict["m180"],
                loss_dict["maux"],
                loss_dict["rgain"],
                loss_dict["rpos"],
                loss_dict["aux_fov"],
                optimizer.param_groups[0]["lr"],
            )
        )

        if cfg.save_latest:
            save_model_state(
                model,
                os.path.join(
                    model_path,
                    "weights_latest.pth",
                ),
            )

        if cfg.save_every_epoch:
            save_model_state(
                model,
                os.path.join(
                    model_path,
                    f"weights_epoch_{epoch:03d}.pth",
                ),
            )

        do_eval = (
            not cfg.skip_eval
            and (
                (
                    epoch % cfg.eval_every_n_epoch == 0
                    and epoch != 0
                )
                or epoch == cfg.epochs
            )
        )

        if do_eval:
            print(
                "\n{}[Evaluate | {} | FoV={}]{}".format(
                    30 * "-",
                    protocol_name,
                    cfg.eval_fov,
                    30 * "-",
                )
            )

            r1_test = evaluate(
                config=cfg,
                model=model,
                reference_dataloader=reference_dataloader_test,
                query_dataloader=query_dataloader_test,
                ranks=[1, 5, 10],
                step_size=1000,
                cleanup=True,
            )

            gc.collect()
            torch.cuda.empty_cache()

            if cfg.sim_sample:
                r1_train, sim_dict = calc_sim(
                    config=cfg,
                    model=model,
                    reference_dataloader=reference_dataloader_train,
                    query_dataloader=query_dataloader_train,
                    ranks=[1, 5, 10],
                    step_size=1000,
                    cleanup=True,
                )

                gc.collect()
                torch.cuda.empty_cache()

            if r1_test > best_score:
                best_score = r1_test

                save_model_state(
                    model,
                    os.path.join(
                        model_path,
                        f"weights_best_e{epoch:03d}_{r1_test:.4f}.pth",
                    ),
                )

                save_model_state(
                    model,
                    os.path.join(
                        model_path,
                        "weights_best.pth",
                    ),
                )

                print("New best R@1:", best_score)

        # Do not spend time on similarity reshuffling in a bounded smoke run.
        if cfg.custom_sampling and cfg.max_steps_per_epoch == 0:
            train_dataloader.dataset.shuffle(
                sim_dict,
                neighbour_select=cfg.neighbour_select,
                neighbour_range=cfg.neighbour_range,
            )

    save_model_state(
        model,
        os.path.join(
            model_path,
            "weights_end.pth",
        ),
    )

    print("Training finished. Best R@1:", best_score)
    print("All outputs saved to:", model_path)


if __name__ == "__main__":
    main()
