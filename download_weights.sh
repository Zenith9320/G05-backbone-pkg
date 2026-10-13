#!/usr/bin/env bash
# ============================================================================
# 下载 backbone 微调所需权重到本地（不入库，checkpoints/ 已在 .gitignore）
#
# 1) 官方 backbone + processor —— 来自公开仓库 OpenGalaxea/G05
#      checkpoints/g05-base/checkpoints/model_state_dict.pt  (~11GB)
#      checkpoints/qwen3_5_2b_base_processor/                 (~22MB)
#    （用 --include 只拉这两个子目录，避免下载全仓 55GB）
#
# 2) 微调后的 ActionCodec 码本 + YAM stats —— 来自私有 model 仓库（codec 微调产物）
#      checkpoints/action_tokenizer_yam.pt
#      data/stats/yam_stats.json
#
# 必需环境变量：
#   HF_TOKEN       HF 读权限 token（勿硬编码/勿提交）
# 可选：
#   BASE_REPO      公开权重仓库，默认 OpenGalaxea/G05
#   CODEC_REPO     微调码本仓库，默认 Zenith6666/yam-actioncodec
#   HF_ENDPOINT    默认官方 https://huggingface.co；国内下载可 export HF_ENDPOINT=https://hf-mirror.com
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

BASE_REPO="${BASE_REPO:-OpenGalaxea/G05}"
CODEC_REPO="${CODEC_REPO:-Zenith6666/yam-actioncodec}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 读权限 token，勿硬编码）}"
export HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-}"

mkdir -p checkpoints data/stats

echo "================================================================"
echo " 1/3 官方 backbone + processor（$BASE_REPO）"
echo "================================================================"
hf download "$BASE_REPO" --repo-type model --local-dir checkpoints \
  --include "g05-base/**" \
  --include "qwen3_5_2b_base_processor/**"

echo "================================================================"
echo " 2/3 微调后 ActionCodec 码本（$CODEC_REPO）"
echo "================================================================"
hf download "$CODEC_REPO" action_tokenizer_yam.pt --repo-type model --local-dir checkpoints

echo "================================================================"
echo " 3/3 YAM 归一化 stats"
echo "================================================================"
hf download "$CODEC_REPO" yam_stats.json --repo-type model --local-dir data/stats

echo ""
echo "[done] 应存在："
ls -la \
  checkpoints/g05-base/checkpoints/model_state_dict.pt \
  checkpoints/qwen3_5_2b_base_processor \
  checkpoints/action_tokenizer_yam.pt \
  data/stats/yam_stats.json
