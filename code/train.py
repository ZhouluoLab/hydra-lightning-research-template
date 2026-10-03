import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.wandb import log_run_config, watch_model


@hydra.main(version_base=None, config_path="../config", config_name="default_config")
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    logger = instantiate(cfg.logger)
    log_run_config(logger, cfg, cfg.get("tracking"))
    watch_model(logger, model, cfg.get("tracking"))

    callbacks = [instantiate(callback) for callback in cfg.callbacks.values()]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer.args)
    trainer.fit(model, datamodule=datamodule)
    if cfg.run_test:
        trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    train()
