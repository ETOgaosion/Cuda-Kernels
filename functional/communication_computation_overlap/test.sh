#!/bin/bash
export MASTER_ADDR=localhost
export MASTER_PORT=7000

torchrun --nnodes=1 --nproc_per_node=2 test.py