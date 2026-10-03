"""Lightning progress displays and per-epoch metric summaries."""

import logging
from numbers import Real

import torch
from pytorch_lightning.callbacks import Callback, TQDMProgressBar


def _number(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) else None


class StableProgressBar(TQDMProgressBar):
    """Show numeric metrics with a fixed width and three decimal places."""

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        return {
            name: f"{number:8.3f}" if (number := _number(value)) is not None else value
            for name, value in metrics.items()
        }


class EpochMetricsPrinter(Callback):
    """Write epoch-level metrics to both the terminal and Hydra's train.log."""

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if name.endswith("_step"):
                continue
            number = _number(value)
            if number is not None:
                display_name = name.removesuffix("_epoch")
                metrics[display_name] = number

        if metrics:
            summary = " | ".join(
                f"{name}={value:.3f}" for name, value in sorted(metrics.items())
            )
            logging.getLogger(__name__).info("Epoch %d | %s", trainer.current_epoch + 1, summary)
