import os
import argparse
import random
import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from rageo.dataset.cvusa import CVUSADatasetEval
from rageo.transforms import get_transforms_val
from rageo.evaluate.cvusa_and_cvact import evaluate
from rageo.model import TimmModel_ConGeo


@dataclass
class Configuration:
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    img_size: int = 384
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    data_folder: str = "./datasets/CVUSA"
    checkpoint_start: str = ""
    fov: float = 0.0
    random_fov: bool = False
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Seeded CVUSA evaluation for ConGeo/SSCL.")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_folder", type=str,
                        default="./datasets/CVUSA")
    parser.add_argument("--fov", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--model", type=str, default="convnext_base.fb_in22k_ft_in1k_384")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--random_fov", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step_size", type=int, default=1000)

    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    print("Loading checkpoint:", checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    clean = {}
    for k, v in state.items():
        clean[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    print("First missing:", missing[:20])
    print("First unexpected:", unexpected[:20])


def main():
    args = parse_args()
    seed_everything(args.seed)

    config = Configuration()
    config.model = args.model
    config.img_size = args.img_size
    config.batch_size = args.batch_size
    config.data_folder = args.data_folder
    config.checkpoint_start = args.checkpoint
    config.fov = args.fov
    config.num_workers = args.num_workers
    config.random_fov = args.random_fov
    config.gpu_ids = tuple(int(x.strip()) for x in args.gpu_ids.split(",") if x.strip() != "")
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n================ CVUSA Seeded Configuration ================")
    print("Model:             ", config.model)
    print("Checkpoint:        ", config.checkpoint_start)
    print("Data folder:       ", config.data_folder)
    print("FoV:               ", config.fov)
    print("Random FoV:        ", config.random_fov)
    print("Seed:              ", args.seed)
    print("Image size:        ", config.img_size)
    print("Batch size:        ", config.batch_size)
    print("Num workers:       ", config.num_workers)
    print("Device:            ", config.device)
    print("GPU ids:           ", config.gpu_ids)
    print("CUDA available:    ", torch.cuda.is_available())
    print("GPU count:         ", torch.cuda.device_count())
    print("============================================================\n")

    model = TimmModel_ConGeo(
        config.model,
        pretrained=False,
        img_size=config.img_size,
        random_fov=config.random_fov,
    )

    load_checkpoint(model, config.checkpoint_start)

    data_config = model.get_config()
    print("Model data config:", data_config)

    mean = data_config["mean"]
    std = data_config["std"]

    image_size_sat = (config.img_size, config.img_size)
    new_width = config.img_size * 2
    new_height = round((224 / 1232) * new_width)
    img_size_ground = (new_height, new_width)

    print("Image Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean:", mean)
    print("Std:", std)

    print("GPUs available:", torch.cuda.device_count())
    valid_gpu_ids = tuple(g for g in config.gpu_ids if g < torch.cuda.device_count())

    if torch.cuda.device_count() > 1 and len(valid_gpu_ids) > 1:
        print("Using DataParallel with gpu_ids:", valid_gpu_ids)
        model = torch.nn.DataParallel(model, device_ids=valid_gpu_ids)
    else:
        print("Use single GPU or CPU.")

    model = model.to(config.device)
    model.eval()

    sat_transforms_val, ground_transforms_val = get_transforms_val(
        image_size_sat,
        img_size_ground,
        mean=mean,
        std=std,
        fov=config.fov,
    )

    reference_dataset_test = CVUSADatasetEval(
        data_folder=config.data_folder,
        split="test",
        img_type="reference",
        transforms=sat_transforms_val,
    )

    query_dataset_test = CVUSADatasetEval(
        data_folder=config.data_folder,
        split="test",
        img_type="query",
        transforms=ground_transforms_val,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    reference_dataloader_test = DataLoader(
        reference_dataset_test,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    top1_percent_rank = max(1, len(reference_dataset_test) // 100)
    ranks = [1, 5, 10, top1_percent_rank]

    print("\nEvaluation ranks:", ranks)
    print("Recall@1% rank:", top1_percent_rank)
    print("Note: if evaluate() prints Recall@{}, it corresponds to Recall@1%.".format(
        top1_percent_rank))

    print("\n{}[{}]{}".format(
        30 * "-",
        f"CVUSA_FoV{config.fov}_Seed{args.seed}",
        30 * "-"
    ))

    evaluate(
        config=config,
        model=model,
        reference_dataloader=reference_dataloader_test,
        query_dataloader=query_dataloader_test,
        ranks=ranks,
        step_size=args.step_size,
        cleanup=True,
    )


if __name__ == "__main__":
    main()
