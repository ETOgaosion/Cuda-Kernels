#!/bin/bash
MASTER_ADDR=localhost
MASTER_PORT=7000

export CUDA_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT test_model.py