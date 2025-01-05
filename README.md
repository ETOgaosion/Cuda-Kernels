# Cuda Learning Notes and Useful Kernels

Include CUDA learning notes and useful CUDA kernels

CUDA Courses and Books:

- [CUDATutorial](https://cuda.keter.top/), go to [CUDATutorial directory](./learning/CUDATutorial/)
- [gpu-mode lectures](https://github.com/gpu-mode/lectures), go to [gpu-mode directory](./learning/gpu-mode/)

Useful CUDA kernels:

- [Occupy GPU SMs and Memory](./functional/occupy_gpu/)

## Environment

All Kernels develop on RTX 3090, with docker container `whatcanyousee/cuda-kernels` with some supplementary packages over `nvcr.io/nvidia/pytorch:24.04-py3`

To start docker environment, refer to [start docker script](./scripts/start_docker.sh)