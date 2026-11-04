#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Formal post-hoc evaluation for RAGeo on VIGOR.

Purpose
-------
Evaluate a trained RAGeo-VIGOR checkpoint without any further training.

This script intentionally reuses the model class, verified VIGOR dataset,
and checkpoint-loading logic from the formal training script:

    train_vigor.py

Supported protocols
-------------------
1. Same-Area / Cross-Area
2. Arbitrary evaluation FoV, e.g. 360 or 90
3. R@1, R@5, R@10, R@1% and Hit_Rate are printed by the original
   congeo.evaluate.vigor.evaluate() implementation.

Example
-------
Same-Area, FoV=360:

python eval_rageo_vigor_formal.py \
    --same_area 1 \
    --fov 360 \
    --checkpoint ./balanced_fov_sscl_vigor/rageo_vigor_same_formal/convnext_base.fb_in22k_ft_in1k_384/20260910_223320/weights_best_e060_24.7433.pth
"""

import os
import gc
import argparse
import importlib.util
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from rageo.transforms import get_transforms_val
from rageo.evaluate.vigor import evaluate


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class EvalConfig:
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    img_size: int = 384

    batch_size_eval: int = 128
    num_workers: int = 4

    normalize_features: bool = True
    verbose: bool = True

    data_folder: str = (
        "./datasets/VIGOR"
    )

    same_area: bool = True
    eval_fov: float = 360.0

    checkpoint: str = ""

    # Keep consistent with formal RAGeo-VIGOR training.
    random_fov: bool = False

    gpu_id: int = 0

    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# 2. Utility
# ============================================================

def str2bool_int(value):
    value = str(value).strip().lower()

    if value in {"1", "true", "yes", "y"}:
        return True

    if value in {"0", "false", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected 0/1 or true/false, got: {value}"
    )


def cleanup_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_formal_training_module(script_path):
    """
    Dynamically load definitions from the formal RAGeo-VIGOR
    training script without executing its main() function.
    """

    script_path = os.path.abspath(script_path)

    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"Formal training script not found:\n{script_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "rageo_vigor_formal_train",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load Python module from:\n{script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "TimmModelBalanced",
        "VigorDatasetEvalVerified",
        "load_resume_checkpoint",
    ]

    for name in required:
        if not hasattr(module, name):
            raise AttributeError(
                f"{script_path} does not contain required "
                f"definition: {name}"
            )

    return module


# ============================================================
# 3. Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Formal RAGeo VIGOR post-hoc evaluation"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RAGeo checkpoint.",
    )

    parser.add_argument(
        "--same_area",
        type=str2bool_int,
        default=True,
        help="1=True for Same-Area; 0=False for Cross-Area.",
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=360.0,
        help="Evaluation FoV, e.g. 360 or 90.",
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        default=EvalConfig.data_folder,
        help="Path to VIGOR_verified.",
    )

    parser.add_argument(
        "--formal_train_script",
        type=str,
        default="train_vigor.py",
        help=(
            "Formal training script whose verified dataset/model "
            "definitions are reused."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=EvalConfig.model,
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )

    return parser.parse_args()


# ============================================================
# 4. Sanity checks
# ============================================================

def check_paths(cfg, train_script):
    if not os.path.isfile(cfg.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{cfg.checkpoint}"
        )

    if not os.path.isdir(cfg.data_folder):
        raise FileNotFoundError(
            f"Dataset folder not found:\n{cfg.data_folder}"
        )

    train_script = os.path.abspath(train_script)

    if not os.path.isfile(train_script):
        raise FileNotFoundError(
            f"Formal training script not found:\n{train_script}"
        )

    # Check VIGOR directory structure relevant to the requested split.
    if cfg.same_area:
        cities = [
            "Chicago",
            "NewYork",
            "SanFrancisco",
            "Seattle",
        ]
    else:
        # Cross-Area TEST cities.
        cities = [
            "Chicago",
            "SanFrancisco",
        ]

    for city in cities:
        city_dir = os.path.join(
            cfg.data_folder,
            city,
        )

        split_dir = os.path.join(
            cfg.data_folder,
            "splits",
            city,
        )

        if not os.path.isdir(city_dir):
            raise FileNotFoundError(
                f"City directory not found:\n{city_dir}"
            )

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found:\n{split_dir}"
            )

    print("[OK] Required paths checked.")


# ============================================================
# 5. Formal evaluation
# ============================================================

@torch.no_grad()
def main():
    args = parse_args()

    cfg = EvalConfig()

    cfg.model = args.model
    cfg.img_size = args.img_size
    cfg.batch_size_eval = args.batch_size_eval
    cfg.num_workers = args.num_workers

    cfg.data_folder = os.path.abspath(args.data_folder)
    cfg.same_area = args.same_area
    cfg.eval_fov = float(args.fov)

    cfg.checkpoint = os.path.abspath(args.checkpoint)

    cfg.gpu_id = args.gpu_id

    if torch.cuda.is_available():
        if cfg.gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU {cfg.gpu_id}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) "
                f"are visible."
            )

        torch.cuda.set_device(cfg.gpu_id)
        cfg.device = f"cuda:{cfg.gpu_id}"
    else:
        cfg.device = "cpu"

    formal_train_script = os.path.abspath(
        args.formal_train_script
    )

    check_paths(
        cfg,
        formal_train_script,
    )

    # --------------------------------------------------------
    # Reuse EXACT formal RAGeo model / verified VIGOR dataset.
    # --------------------------------------------------------

    formal = load_formal_training_module(
        formal_train_script
    )

    TimmModelBalanced = formal.TimmModelBalanced
    VigorDatasetEvalVerified = (
        formal.VigorDatasetEvalVerified
    )
    load_resume_checkpoint = (
        formal.load_resume_checkpoint
    )

    protocol_name = (
        "Same-Area"
        if cfg.same_area
        else "Cross-Area"
    )

    print()
    print("=" * 80)
    print("RAGeo VIGOR Formal Post-hoc Evaluation")
    print("=" * 80)

    print("Protocol:", protocol_name)
    print("FoV:", cfg.eval_fov)
    print("Model:", cfg.model)
    print("Checkpoint:", cfg.checkpoint)
    print("Dataset:", cfg.data_folder)
    print("Formal train script:", formal_train_script)
    print("Device:", cfg.device)
    print("Batch size:", cfg.batch_size_eval)
    print("Num workers:", cfg.num_workers)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TimmModelBalanced(
        cfg.model,
        pretrained=False,
        img_size=cfg.img_size,
        random_fov=cfg.random_fov,
    )

    load_resume_checkpoint(
        model,
        cfg.checkpoint,
    )

    data_config = model.get_config()

    mean = data_config["mean"]
    std = data_config["std"]

    print("Data config:", data_config)

    # Same preprocessing as formal training:
    # satellite = 384 x 384
    # ground    = 384 x 768
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

    print("Image Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean:", mean)
    print("Std:", std)

    model = model.to(cfg.device)
    model.eval()

    # --------------------------------------------------------
    # Evaluation transforms
    # --------------------------------------------------------

    sat_val, ground_val = get_transforms_val(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std,
        fov=cfg.eval_fov,
    )

    # --------------------------------------------------------
    # Verified VIGOR evaluation datasets
    # --------------------------------------------------------

    reference_dataset_test = (
        VigorDatasetEvalVerified(
            cfg.data_folder,
            split="test",
            img_type="reference",
            same_area=cfg.same_area,
            transforms=sat_val,
        )
    )

    query_dataset_test = (
        VigorDatasetEvalVerified(
            cfg.data_folder,
            split="test",
            img_type="query",
            same_area=cfg.same_area,
            transforms=ground_val,
        )
    )

    reference_dataloader_test = DataLoader(
        reference_dataset_test,
        batch_size=cfg.batch_size_eval,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            cfg.num_workers > 0
        ),
    )

    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=cfg.batch_size_eval,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            cfg.num_workers > 0
        ),
    )

    print()
    print(
        "Reference Images Test:",
        len(reference_dataset_test),
    )

    print(
        "Query Images Test:",
        len(query_dataset_test),
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print(
        "{}[Evaluate | {} | FoV={}]{}".format(
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

    print()
    print("=" * 80)
    print("Evaluation finished")
    print("=" * 80)

    print("Protocol:", protocol_name)
    print("FoV:", cfg.eval_fov)
    print("Checkpoint:", cfg.checkpoint)
    print("Returned R@1:", r1_test)

    print()
    print(
        "NOTE: Full metrics "
        "(R@1 / R@5 / R@10 / R@1% / Hit_Rate) "
        "are printed above by congeo.evaluate.vigor.evaluate()."
    )

    del model
    cleanup_cuda()


if __name__ == "__main__":
    main()