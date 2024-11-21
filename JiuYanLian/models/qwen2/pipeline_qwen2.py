# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import copy
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage

from JiuYanLian.config_manager import JobConfig

from JiuYanLian.logging import logger
# from JiuYanLian.models.llama import ModelArgs
from transformers import AutoConfig as ModelArgs
from JiuYanLian.parallelisms.parallel_dims import ParallelDims
from JiuYanLian.parallelisms.pipelining_utils import (
    build_pipeline_schedule,
    generate_split_points,
    stage_ids_this_rank,
)


DeviceType = Union[int, str, torch.device]


def pipeline_parallelize(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: ModelArgs,
    loss_fn: Callable[..., torch.Tensor],
):
    stages, models = pipeline_manual_split(
        model, pp_mesh, parallel_dims, job_config, device, model_config
    )

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    return pp_schedule, models


def pipeline_manual_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: ModelArgs,
):
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    microbatches = (
        job_config.experimental.pipeline_parallel_microbatches or parallel_dims.pp
    )
    splits = (
        job_config.experimental.pipeline_parallel_split_points
        or generate_split_points(job_config, parallel_dims.pp, model_config)
    )

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.model.embed_tokens = nn.Identity()
            # model.model.rotary_emb = None

        # drop_layers = start_layer is not None
        # for name in list(model.model.layers.keys()):
        # for layer_index in reversed(range(len(model.model.layers))):
        #     # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
        #     drop_layers = False
        #     if start_layer is not None and layer_index < start_layer:
        #         drop_layers = True
        #     if stop_layer is not None and layer_index >= stop_layer:
        #         drop_layers = True
        #     if drop_layers:
        #         del model.model.layers[layer_index]
        if start_layer is None:
            start_layer = 0 
        if stop_layer is None:
            stop_layer = len(model.model.layers) 
        model.model.layers = model.model.layers[start_layer:stop_layer]    

        if not is_last:
            model.model.norm = nn.Identity()
            model.lm_head = nn.Identity()

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    stages = []
    models = []
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop"):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models