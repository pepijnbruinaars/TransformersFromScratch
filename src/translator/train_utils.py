import os
import glob
import json
import random
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_splits(run_folder: str, train_raw: Any, validation_raw: Any, test_raw: Any) -> None:
    ensure_dir(run_folder)
    path = os.path.join(run_folder, "splits.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"train": train_raw, "validation": validation_raw, "test": test_raw}, f, ensure_ascii=False)


def load_splits(run_folder: str):
    path = os.path.join(run_folder, "splits.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["train"], data["validation"], data["test"]


def save_checkpoint(path: str, transformer: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, epoch: int, extra: dict | None = None) -> None:
    ckpt = {
        "model_state_dict": transformer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str | None = None) -> dict:
    map_loc = None if device is None else device
    return torch.load(path, map_location=map_loc)


def restore_rng_states(ckpt: dict) -> None:
    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "python_rng_state" in ckpt:
        random.setstate(ckpt["python_rng_state"])


def find_latest_run(base_dir: str = "models/transformer") -> str | None:
    if not os.path.isdir(base_dir):
        return None
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def find_latest_checkpoint(run_folder: str) -> str | None:
    patterns = [os.path.join(run_folder, "transformer_epoch_*.pt"), os.path.join(run_folder, "transformer_best.pt"), os.path.join(run_folder, "transformer_final.pt")]
    epoch_files = glob.glob(patterns[0])
    if epoch_files:
        def epoch_num(p: str) -> int:
            name = os.path.basename(p)
            try:
                return int(name.split("_")[-1].split(".")[0])
            except Exception:
                return -1

        return max(epoch_files, key=epoch_num)
    for p in patterns[1:]:
        files = glob.glob(p)
        if files:
            return files[0]
    return None
