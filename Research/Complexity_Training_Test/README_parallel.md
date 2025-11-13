# Multi-GPU Training with train_parallel.py

This script supports both single-GPU and multi-GPU training using PyTorch's DistributedDataParallel (DDP).

## Single GPU Training

Simply run the script normally:

```bash
python train_parallel.py
```

## Multi-GPU Training

Use `torchrun` to launch the script across multiple GPUs:

```bash
# For 2 GPUs
torchrun --nprocs_per_node=2 train_parallel.py

# For all available GPUs
torchrun --nprocs_per_node=$(nvidia-smi -L | wc -l) train_parallel.py
```
