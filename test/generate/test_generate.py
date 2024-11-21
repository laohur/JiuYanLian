# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
import time
from pathlib import Path

from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan import utils

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.generation import apply_torchchat_tp, generate


@record
def example_generate(
    config_path: str,
    checkpoint_path: str,
    prompt: str,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
):
    init_logger()
    color = utils.Color

    # Load configuration from toml file
    config = JobConfig()
    config.parse_args([f"--job.config_file={config_path}"])
    config._validate_config()

    utils.set_determinism(seed)

    if seed is None:
        logger.info("Deterministic sampling off")
    else:
        logger.info(f"Deterministic sampling on. Using seed: {seed}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    gpu_memory_monitor = build_gpu_memory_monitor()

    model_name = config.model.name

    # Init distributed env
    if world_size > 1:
        utils.init_distributed(config)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=world_size,
            pp=1,
            world_size=world_size,
            enable_loss_parallel=False,
        )
        # Build world mesh for parallelism
        world_mesh = parallel_dims.build_mesh(device_type="cuda")

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    # Tokenizer setup
    tokenizer = build_tokenizer(
        model_name_to_tokenizer[model_name], config.model.tokenizer_path
    )

    model_config = models_config[model_name][config.model.flavor]
    model_config.norm_type = config.model.norm_type
    model_config.max_seq_len = config.training.seq_len
    model_config.vocab_size = tokenizer.n_words

    model_cls = model_name_to_cls[model_name]
    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls.from_model_args(model_config)

    if world_size > 1:
        use_torchchat_tp = False
        if use_torchchat_tp:
            apply_torchchat_tp(model, world_mesh["tp"])  # Working
        else:
            models_parallelize_fns[model_name](model, world_mesh, parallel_dims, config)

    # materalize model
    model.to_empty(device="cuda")
    model.eval()

    state_dict = {"model": model.state_dict()}

    # Checkpoint Loading
    begin = time.monotonic()
    logger.info(f"Loading chkpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    # Tokenize prompt and repeat batch_size times
    input_ids = (
        (
            torch.tensor(
                tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.long
            )
            .view(1, -1)
            .repeat(batch_size, 1)
        )
        .cuda()
        .detach()
    )

    gpu_memory_monitor.reset_peak_stats()

    # Run generation
    t0 = time.monotonic()
    responses = generate(
        model,
        input_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
    )
    t1 = time.monotonic()
    elapsed_sec = t1 - t0

    # Post process
    B, T = responses.size()  # B: batch_size, T: total seq length
    input_n_tokens = input_ids.size(1)
    generated_n_tokens = T - input_n_tokens  # == max_new_tokens

    if local_rank == 0:
        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")

        r, b = color.red, color.blue

        output_data = {
            "metadata": {},
            "responses": [],
        }

        for i, tokens in enumerate(responses):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()

            input_text = tokenizer.decode(inp_tok)
            output_text = tokenizer.decode(out_tok)

            _data = {
                "response_idx": i,
                "input_text": input_text,
                "output_text": output_text,
            }
            output_data["responses"].append(_data)

            logger.info(f"{r}\n{input_text}{b}{output_text}\n{color.reset}")

        gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
        output_data["metadata"] = {
            "generated_n_tokens": generated_n_tokens,
            "input_n_tokens": input_n_tokens,
            "generation_time_sec": elapsed_sec,
            "tokens_per_sec": (B * T) / elapsed_sec,
            "batch_size": B,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
            "memory/max_active(%)": gpu_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
            "memory/num_ooms": gpu_mem_stats.num_ooms,
            "world_size": world_size,
            "torch_version": torch.__version__,
        }
        print(json.dumps(output_data, indent=4))


def load_prompt(prompt):
    prompt_path = Path(prompt)

    if prompt_path.exists():
        if prompt_path.is_file():
            try:
                content = prompt_path.read_text()
                if content:  # Ensure the file is not empty
                    return content
                print("Error: Prompt file is empty.")
            except IOError as e:
                print(f"Error: Unable to read file '{prompt_path}'. {e}")
        else:
            print(f"Error: Path '{prompt}' is not a file.")
    # If not empty, streat as a string
    elif prompt:
        return prompt

    print("Error: Provided prompt is empty or file does not exist")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default is 1.0",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max number of tokens to generate. Default is 32",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of samples to run in batch"
    )
    parser.add_argument(
        "--top_k", type=int, help="Prune to select from top_k probabilities. Optional"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello! How are",
        help="Input prompt for generation, either as a string or a path to a .txt file",
    )

    args = parser.parse_args()
    prompt_text = load_prompt(args.prompt)

    example_generate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        prompt=prompt_text,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()