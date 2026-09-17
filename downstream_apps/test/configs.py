"""
Task-specific configuration for the wave-classification app.

Everything generic — paths, channels, temporal sampling, S3 settings, the model and
LoRA configs, the training and logging sections, and ``load_config()`` itself — lives in
``workshop_infrastructure/configs.py``. This file holds only what is specific to *this*
task: the wave catalog and how its events are aligned to the Surya index.

**This is the pattern to copy when you fork the template.** Subclass ``DataConfig`` with
your task's fields, then bind ``load_config`` to it. You never maintain a copy of the
base config.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import ClassVar

from workshop_infrastructure.configs import (  # re-exported for convenience
    DataConfig,
    LoraAdapterConfig,
    ModelConfig,
    OutputConfig,
    TimeEmbeddingConfig,
    TrainingConfig,
    load_config,
)


@dataclass
class WaveDataConfig(DataConfig):
    """DataConfig plus the wave-catalog alignment settings used by ``waveDSDataset``.

    These four keys are what makes this app's ``data:`` section different from any other
    downstream task's. Swap them for your own when you fork.
    """
    # Path to the label catalog (relative paths resolve against the config file's dir).
    wave_index_path: str = ""
    # Column in the catalog holding the event timestamp.
    ds_time_column: str = "start_time"
    # Max allowed gap when matching catalog events to Surya timesteps.
    ds_time_tolerance: str = "4d"
    # "forward" uses the solar state *before* the event (causal prediction).
    ds_match_direction: str = "forward"
    # Column in the catalog holding the label (e.g. "wave" / "no wave").
    ds_class_column: str = ""

    # wave_index_path is a path, so it must join the base class's list to get the same
    # relative-to-the-config-file resolution. Extend this whenever you add a path field.
    PATH_FIELDS: ClassVar[tuple[str, ...]] = DataConfig.PATH_FIELDS + ("wave_index_path",)


# The app's entry point. Identical to load_config() except that the data: section is
# parsed into WaveDataConfig, so the four keys above are recognized instead of rejected.
load_wave_config = partial(load_config, data_cls=WaveDataConfig)


__all__ = [
    "WaveDataConfig",
    "load_wave_config",
    # Re-exports so app code can import everything config-related from one place.
    "DataConfig",
    "OutputConfig",
    "TrainingConfig",
    "ModelConfig",
    "LoraAdapterConfig",
    "TimeEmbeddingConfig",
    "load_config",
]
