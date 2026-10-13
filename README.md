# G05-backbone-pkg

G0.5 复现第二步 —— **backbone（VLA 全模型）微调**代码。从上游
[OpenGalaxea/GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) 抽取微调 + 评测所需
部分，供算力平台 clone 后直接微调。

> 配套：ActionCodec 码本微调在 `ActionCodec-finetune-pkg`（产物
> `action_tokenizer_yam.pt` 已上传 HF，本仓库 `download_weights.sh` 会拉取）。

## 目录

```
├── scripts/
│   ├── finetune.py               # 训练入口
│   ├── eval_open_loop.py         # open-loop 评测入口
│   ├── run/finetune.sh           # 训练 launcher：bash scripts/run/finetune.sh <n_gpu> yam ...
│   ├── run/eval_open_loop.sh     # 评测 launcher
│   └── utils/                    # finetune.py 依赖（metric/preflight/eval_snapshot/train_eval）
├── src/g05/                      # 模型 + 数据 + processor 全部源码
├── configs/                      # train.yaml / model/g05.yaml / task/yam.yaml / data/yam.yaml
├── pyproject.toml + uv.lock      # uv 依赖（Python 3.10 + cu128 torch）
├── setup_env.sh                  # 装 uv + uv sync
├── download_weights.sh           # 下 backbone + processor + codec + stats
├── download_dataset.sh           # 下 data/yam/TASK-YAM-*_lerobot
└── upload_output.sh              # 上传微调产物到 HF
```

## 算力平台 runbook

```bash
# 0. 克隆 + 装环境（uv sync）
git clone https://github.com/Zenith9320/G05-backbone-pkg && cd G05-backbone-pkg
bash setup_env.sh

# 1. 设 token + 输出目录
export HF_TOKEN="hf_xxxxxxxx"                      # 读写权限 token
export G05_OUTPUT_DIR=/data/output/g05_yam         # 训练输出目录（平台持久盘）

# 2. 下载权重 + 数据集（ckpt / 训练集都不用本地准备，脚本从 HF 拉）
bash download_weights.sh        # g05-base + processor + action_tokenizer_yam.pt + yam_stats.json
bash download_dataset.sh        # data/yam/TASK-YAM-*_lerobot

# 3. 四级验证（逐级通过再正式训练）
source .venv/bin/activate
bash scripts/run/finetune.sh 1 yam --dry-run --max_datasets 1
bash scripts/run/finetune.sh 1 yam --test --max_datasets 1 model.pretrained_ckpt=null model.max_steps=10
bash scripts/run/finetune.sh 1 yam --overfit_batch 5 model.max_steps=500

# 4. 正式训练（GPU 数按平台定；wandb 离线）
bash scripts/run/finetune.sh 8 yam logger.mode=offline

# 5. open-loop 评测（指向产出 ckpt）
bash scripts/run/eval_open_loop.sh \
  --ckpt_path $G05_OUTPUT_DIR/yam/<exp>/last.pt \
  --task_config configs/task/yam.yaml

# 6. 上传产物（防实例结束丢失）
EXP_DIR=$G05_OUTPUT_DIR/yam/<exp> bash upload_output.sh
```

### 通过标准

- 正式训练日志：`Loaded N/N`、`Missing (rand init): 0`，loss 稳定下降。
- 输出目录 `$G05_OUTPUT_DIR/yam/<exp>/`：`checkpoints/`、`dataset_stats.json`、
  `.hydra/config.yaml`、`action_tokenizer.pt`。

## 权重路径对齐（脚本已保证）

| 文件 | 来源 | 配置引用 |
|---|---|---|
| `checkpoints/g05-base/checkpoints/model_state_dict.pt` | `OpenGalaxea/G05` | `configs/model/g05.yaml` → `pretrained_ckpt` |
| `checkpoints/qwen3_5_2b_base_processor/` | `OpenGalaxea/G05` | `configs/model/g05.yaml` → `hf_processor_path` |
| `checkpoints/action_tokenizer_yam.pt` | `Zenith6666/yam-actioncodec` | `configs/task/yam.yaml` → `tokenizer.vq_config.ckpt_dir` |
| `data/stats/yam_stats.json` | `Zenith6666/yam-actioncodec` | `configs/task/yam.yaml` → `datastatics_path` |
| `data/yam/TASK-YAM-XXXX_lerobot` | `Zenith6666/yam-lerobot` | `configs/data/yam.yaml` → `dataset_dirs` |

## 说明

1. **任务指令文本**：backbone 是 VLA，需要每条 episode 的真实自然语言指令。数据集已按
   `ActionCodec-finetune-pkg/tools/yam/tasks_yam.json` 的真实英文指令转换，parquet `task` 字段已就绪。
2. **全参微调**：`configs/task/yam.yaml` 已配 `vla_training_strategy: vla-full-train`
   + differential LR（`vision_lr_multiplier: 0.1` / `backbone_lr_multiplier: 1.0`）。
