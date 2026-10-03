"""W&B experiment configuration and optional model monitoring."""

import os

from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger


def watch_model(logger, model, settings=None):
    """Optionally watch the model on global rank zero only."""
    settings = settings or {}
    if not settings.get("watch_model", True):
        return
    if isinstance(logger, WandbLogger) and os.environ.get("RANK", "0") == "0":
        logger.watch(
            model,
            log=settings.get("watch_log", "gradients"),
            log_freq=settings.get("watch_log_freq", 100),
            log_graph=settings.get("watch_graph", True),
        )


def log_run_config(logger, cfg, settings=None):
    """Record the selected training configuration, excluding arbitrary user secrets."""
    settings = settings or {}
    if not settings.get("log_config", False):
        return
    if not isinstance(logger, WandbLogger) or os.environ.get("RANK", "0") != "0":
        return

    metadata = {key: cfg.get(key) for key in ("project", "name", "seed")}
    for key in ("model", "data", "trainer"):
        metadata[key] = OmegaConf.to_container(cfg[key], resolve=True)
    logger.log_hyperparams(metadata)
