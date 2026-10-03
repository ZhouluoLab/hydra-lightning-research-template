# Hydra 配置说明

`default_config.yaml` 是 MNIST 流水线检查入口；`project_config.yaml` 是自定义任务入口，目前仍复用 MNIST 示例。它们的 `defaults` 从 `model/`、`data/`、`logger/`、`callbacks/`、`trainer/` 各选择一份 YAML。`_self_` 让入口中的值最后合并。子文件的 `# @package model` 等指令决定最终配置节点；`_target_` 告诉 `hydra.utils.instantiate()` 应实例化哪个 Python 类。

| 配置 | 作用 | 如何替换 |
| --- | --- | --- |
| `model/resnet18.yaml` | 默认 LightningModule、类别数、学习率 | 新增 `model/my_model.yaml`，运行时指定 `model=my_model` |
| `data/default_data.yaml` | MNIST DataModule、每步 batch、数据量上限 | 新增 `data/my_data.yaml`，运行时指定 `data=my_data` |
| `logger/wandb_offline.yaml` | 默认本地 W&B 记录 | `logger=wandb_online` 实时上传 |
| `callbacks/default_callbacks.yaml` | 三位小数 tqdm、epoch 汇总、提前停止与 checkpoint | 新增回调 YAML，或覆盖回调参数 |
| `trainer/default_trainer.yaml` | Lightning Trainer 的 epochs、设备等 | `trainer.args.max_epochs=3` 等普通覆盖 |

`model`、`data`、`logger` 和每个 `callbacks` 项由 `code/train.py` 调用 `instantiate()`；`trainer.args` 展开传给 `pl.Trainer`。`trainer.metric` / `trainer.mode` 被 EarlyStopping 和 ModelCheckpoint 引用，默认监控 `val_loss` 且越小越好。改监控准确率时同步设置 `trainer.metric=val_acc trainer.mode=max`，模型需记录该名字。

`data.train_samples`、`data.val_samples`、`data.test_samples` 中的 `null` 是使用完整划分；填正整数用于短测，不改变 MNIST 的 55,000/5,000 分割比例。`data.batch_size` 是每个训练进程的 batch 大小。`seed` 控制 Lightning 随机种子，`data.split_seed` 单独固定 MNIST 划分。`run_test` 控制训练结束后是否使用最佳 checkpoint 执行测试。

入口文件中的 `hydra.run.dir: outputs/${now:...}` 决定每次实验目录；`hydra.job.chdir: true` 切换训练工作目录，因此相对的 `checkpoints` 路径位于本次实验目录。`logger.project/name/tags` 分别是 W&B 项目、run 显示名和标签；科研入口的 `tracking.log_config` 控制是否额外记录解析后的模型、数据与 Trainer 配置。

先看结果再运行：

```bash
python code/train.py --cfg job --resolve
python code/train.py --config-name project_config --cfg job --resolve \
  model=resnet18 data=default_data trainer.args.max_epochs=3
```

组选择写 `model=my_model`，已有字段改值写 `trainer.args.max_epochs=3`，给 Trainer 新增参数写 `+trainer.args.limit_train_batches=1`。详见 [Hydra 官方 override 语法](https://hydra.cc/docs/advanced/override_grammar/basic/)。
