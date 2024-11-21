#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/qwen2_7b.toml"}

# overrides=""
# if [ $# -ne 0 ]; then
#     overrides="$*"
# fi

# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
# torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
# --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
# train.py --job.config_file ${CONFIG_FILE} $overrides

# export CUDA_VISIABLE_DEVICES=3,5,6,7
# export NGPU=4
# export LOG_RANK=0,1
# export CONFIG_FILE="./train_configs/qwen2_7b.toml"
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
scripts/train_qwen.py --job.config_file ${CONFIG_FILE} $overrides
