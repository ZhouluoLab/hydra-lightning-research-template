# Hydra + PyTorch Lightning 深度学习实验模板

一份可从 MNIST 快速起步、再替换为自己研究任务的训练骨架。Hydra 负责组合模型、数据、日志、回调和 Trainer 配置；Lightning 负责训练/验证/测试与 checkpoint；W&B 默认离线记录。`code/utils/` 保留数据加载、进度和 W&B 等可复用工具，今后可直接在其中扩展。

这是公开的通用项目模板。可在 GitHub 仓库页面点击 **Use this template → Create a new repository**，为自己的科研项目创建独立仓库，并自行选择新仓库的公开或私有状态。模板更新不会自动覆盖你的项目。

本模板基于 [zerebom/hydra-pl-wandb-sample-project](https://github.com/zerebom/hydra-pl-wandb-sample-project) 的项目结构，经本地迭代整理了训练代码、MNIST 示例和可复用工具。

本模板**不绑定任何主机、NFS 路径、SSH 地址或网卡**。默认使用本机单设备；原仓库的三机四卡脚本不在这里。MNIST + ResNet-18 是流水线示例，不是科研模型必须遵循的结构。

## 安装

使用 Python 3.10+ 的独立环境。先从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装适合本机 CPU/CUDA/ROCm 的 `torch` 与 `torchvision`，再执行：

```bash
python -m pip install -r requirements.txt
python -c 'import torch, torchvision, hydra, pytorch_lightning, wandb; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
```

W&B 默认离线，不必先登录。首次运行 MNIST 时 torchvision 会下载数据至 `data/mnist/`。

## 五分钟验证

在仓库根目录执行。`--cfg job --resolve` 只查看 Hydra 合成配置，不开始训练：

```bash
python code/train.py --cfg job --resolve
```

用少量样本、一个 epoch 在 CPU 上检测数据、模型、logger、checkpoint 与测试链路：

```bash
python code/train.py trainer.args.accelerator=cpu trainer.args.max_epochs=1 \
  data.train_samples=128 data.val_samples=64 data.test_samples=64 data.num_workers=0 \
  +trainer.args.limit_train_batches=1 +trainer.args.limit_val_batches=1 \
  +trainer.args.limit_test_batches=1
```

完整 MNIST 示例：

```bash
python code/train.py
```

默认训练集为 55,000 张、验证集为 5,000 张，另有 10,000 张测试图像。`*_samples: null` 表示不裁剪；验证集由 [MNISTDataModule](code/utils/mnist.py) 固定种子划分。`accelerator=auto` 会使用本机可用设备；若只想用 CPU，追加 `trainer.args.accelerator=cpu`。

## 开始自己的项目

`default_config.yaml` 永远保留作 MNIST 健康检查。`project_config.yaml` 目前同样选择 MNIST + ResNet-18，但有独立实验名称与标签。新增自己的模块和配置：

```text
code/networks/my_model.py       自己的 LightningModule
code/utils/my_data.py           自己的 LightningDataModule
config/model/my_model.yaml      模型构造参数
config/data/my_data.yaml        数据构造参数
```

模型实现 `training_step`、`validation_step`、按需 `test_step` 与 `configure_optimizers`；数据模块实现 `setup` 及 train/val/test dataloader。在验证步骤记录要监控的指标（默认 `val_loss`），例如 `self.log("val_loss", loss, on_epoch=True, sync_dist=True)`。没有测试集时设置 `run_test=false`。

```bash
python code/train.py --config-name project_config --cfg job --resolve model=my_model data=my_data
python code/train.py --config-name project_config model=my_model data=my_data \
  project=my-research name=baseline
```

确认新模块可运行后，只修改 `project_config.yaml` 的 `defaults`，以后便不必每次写 `model=` / `data=`，默认 MNIST 入口仍不变。自定义数据可放 `data/<项目名>/`；数据模块应使用 `hydra.utils.to_absolute_path()` 或绝对路径，因为 Hydra 会切换到本次输出目录。详见 [配置说明](config/README.md)。

## 产物与 W&B

每次训练由 Hydra 建立 `outputs/<时间>/`：`.hydra/` 留存配置和命令行覆盖项，`train.log` 留存每轮指标，`checkpoints/` 保存指标最优两份及 `last.ckpt`，`wandb/` 保存离线记录。`run_test: true` 会在训练后加载最佳 checkpoint 测试。

需要上传离线实验时，先在联网环境运行 `wandb login`，再对具体目录执行 `wandb sync outputs/<时间>/wandb/offline-run-*`。这不会自动上传 `checkpoints/` 中的模型。训练时实时上传可用 `logger=wandb_online`，并确保运行机器已登录且联网；需要上传模型 artifact 时另加 `logger.log_model=true`。不要把 API token 或密码写进配置。

源码、配置和文档纳入 Git；数据、权重和 `outputs/` 默认忽略。多机 DDP 需要针对实际调度器、共享存储和网络另配启动方式，这份模板不假定固定集群。PyTorch Lightning 的设备与策略选项见 [官方 Trainer 文档](https://lightning.ai/docs/pytorch/stable/common/trainer.html)。
