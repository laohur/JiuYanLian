# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# from torchtitan.models.llama.model import ModelArgs, Transformer
from transformers import  Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer
from .modeling_qwen2 import Qwen2ForCausalLM4PP, Qwen2ForCausalLM4TP
from .parallelize_qwen2 import parallelize, apply_tp, apply_ac, apply_compile, apply_fsdp, apply_ddp
from .pipeline_qwen2 import pipeline_parallelize