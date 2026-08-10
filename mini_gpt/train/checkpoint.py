"""Checkpoint + resume.

A checkpoint saves the model weights, optimizer(s) state, scheduler state, the
data-sampler position, the step counter, and RNG state, so a killed overnight
run resumes without replaying tokens and produces the same trajectory as an
uninterrupted run. The data stream's state (sampler position) is saved
alongside the trainer so resume does not repeat or skip windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_gpt.train.loop import DataStream, Trainer


def save_checkpoint(
    path: str | Path,
    trainer: Trainer,
    data: DataStream | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "trainer": trainer.state_dict(),
        "data": data.state_dict() if data is not None else None,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "extra": extra or {},
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    trainer: Trainer,
    data: DataStream | None = None,
) -> dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    trainer.load_state_dict(payload["trainer"])
    if data is not None and payload.get("data") is not None:
        data.load_state_dict(payload["data"])
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    if torch.cuda.is_available() and "cuda_rng" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload.get("extra", {})
