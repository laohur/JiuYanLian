

# JiuYanLian

`JiuYanLian` as a powerfull and easy LLM trainer, is currently in a pre-release state and under extensive development. 


Our guiding principles when building `JiuYanLian`:

* easy to understand, use and extend for different training purposes. [transformers](https://github.com/huggingface/transformers) integrated.
* powerfull: Minimal changes to the model code when applying 1D, 2D, or (soon) 3D Parallel. [torchtitan](https://github.com/pytorch/torchtitan).
* support Llama, Qwen, ...

### Dive into the code

You may want to see how the model is defined or how parallelism techniques are applied. For a guided tour, see these files first:
* [scripts/train_qwen.py](scripts/train_qwen.py) - the main training loop and high-level setup code
* [JiuYanLian/models/qwen2/__init__.py](JiuYanLian/models/qwen2/__init__.py) - helpers for applying Data Parallel, Tensor Parallel, activation checkpointing, and `torch.compile` to the qwen2 model
* [tool/convert_hf_to_dcp.py](tool/convert_hf_to_dcp.py) - utils for convert distributed checkpoints from huggingface transformers

### Key features available

1. [FSDP2](docs/fsdp.md) with per param sharding
2. [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) (including [async TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487))
3. Selective layer and operator activation checkpointing
4. [Distributed checkpointing](https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250) (including async checkpointing)
5. Checkpointable data-loading, with the C4 dataset pre-configured (144M entries)
6. Loss, GPU memory, tokens-per-second, and MFU displayed and logged via [TensorBoard](#tensorboard)
7. Learning rate scheduler, meta-init, optional Fused RMSNorm
8. [Float8](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323) support ([how-to](docs/float8.md))
9. `torch.compile` support
10. DDP and HSDP
11. All options easily configured via [toml files](train_configs/)
12. [Interoperable checkpoints](docs/checkpoint.md) which can be loaded directly into [`torchtune`](https://github.com/pytorch/torchtune) for fine-tuning
13. Debugging tools including CPU/GPU profiling, [memory profiling](docs/memory_profiler.md), [Flight Recorder](#debugging), etc.

We report our [Performance](docs/performance.md) verified on 64/128 GPUs.


### Coming soon

- Context Parallel
- Llama


## Installation

```bash
git clone https://github.com/laohur/JiuYanLian
cd JiuYanLian
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 # or cu118
pip install -e .
```

### Start a training run
Llama 3 8B model locally on 8 GPUs

```bash
cd scripts
CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh
```


## TensorBoard

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Make sure `metrics.enable_tensorboard` option is set to true in model training (either from a .toml file or from CLI).

2. Set up SSH tunneling, by running the following from local CLI
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. Inside the SSH tunnel that logged into the remote server, go to the torchtitan repo, and start the TensorBoard backend
```
tensorboard --logdir=./outputs/tb
```

4. In the local web browser, go to the URL it provides OR to http://localhost:6006/.


## Multi-Node Training
For training on ParallelCluster/Slurm type configurations, you can use the `multinode_trainer.slurm` file to submit your sbatch job.

To get started adjust the number of nodes and GPUs
```
#SBATCH --ntasks=2
#SBATCH --nodes=2
```

Then start a run where `nnodes` is your total node count, matching the sbatch node count above.

```
srun torchrun --nnodes 2
```

If your gpu count per node is not 8, adjust:

```--nproc_per_node```

 in the torchrun command and

```#SBATCH --gpus-per-task```

in the SBATCH command section.


## Debugging
### Troubleshooting jobs that timeout
If you encounter jobs that timeout, you'll need to debug them to identify the root cause. To help with this process, we've enabled Flight Recorder, a tool that continuously collects diagnostic information about your jobs.
When a job times out, Flight Recorder automatically generates dump files on every rank containing valuable debugging data. You can find these dump files in the `job.dump_folder` directory.
To learn how to analyze and diagnose issues using these logs, follow our step-by-step tutorial [link](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html).


## License

This code is made available under [BSD 3 license](./LICENSE). However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models, data, etc.
