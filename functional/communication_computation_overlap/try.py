import torch
import torch.nn as nn
import torch.distributed as dist
import os
import cupy as cp
import time

# Initialize distributed environment
print(f"Rank {os.getenv('RANK')} in {os.getenv('WORLD_SIZE')} with {os.getenv('MASTER_ADDR')}:{os.getenv('MASTER_PORT')} initializing...")
dist.init_process_group(backend='nccl', init_method=f"tcp://{os.getenv('MASTER_ADDR')}:{os.getenv('MASTER_PORT')}", rank=int(os.getenv('RANK')), world_size=int(os.getenv('WORLD_SIZE')))
print(f"Rank {dist.get_rank()} initialized.")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
cp.cuda.runtime.setDevice(rank)

cuda_kernel = cp.RawKernel(r'''
extern "C" __global__
void matmul_kernel(float *A, float *B, float *C, int M, int N, int K, int iters) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int range = 0; range < iters; range++) {
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
        }
        C[row * N + col] = sum;
    }
}
''', 'matmul_kernel')

def communication(iter_range, comm_stream):
    # Allocate tensors
    size = 1024 * 1024 * 1024  # 16M elements (~64MB for fp32)
    tensor_send = torch.ones(size, dtype=torch.float32, device=device)
    tensor_recv = torch.empty(size, dtype=torch.float32, device=device)
    """Conducts P2P communication using full bandwidth."""
    peer = (rank + 1) % world_size  # Simple ring communication
    reqs = []  # 保存所有通信请求

    with torch.cuda.stream(comm_stream):
        for _ in range(iter_range):
            send_op = dist.P2POp(dist.isend, tensor_send, (rank + 1)%world_size)
            recv_op = dist.P2POp(dist.irecv, tensor_recv, (rank + 1)%world_size)
            reqs = dist.batch_isend_irecv([send_op, recv_op])

    for req in reqs:
        req.wait()

def run(iters=10):
    s1 = torch.cuda.Stream(device=device)
    s2 = torch.cuda.Stream(device=device)
    N = 1024 * 4
    x = torch.rand(size=(N, N)).to(device)
    w1 = torch.rand(size=(N, N)).to(device)
    w2 = torch.rand(size=(N, N)).to(device)
    o1 = torch.empty(size=(N, N)).to(device)
    o2 = torch.empty(size=(N, N)).to(device)
    threads_per_block = int(32)
    num_blocks = (N + threads_per_block - 1) // threads_per_block
    
    for i in range(iters):
        torch.cuda.nvtx.range_push('iter{}'.format(i))

        with torch.cuda.stream(s1):
            # out1 = x.matmul(w1)
            cuda_kernel((num_blocks,), (threads_per_block,), (x.data_ptr(), w1.data_ptr(), o1.data_ptr(), N, N, N, 1000 * 3))
    
        # with torch.cuda.stream(s2):
        #     # out2 = x.matmul(w2)
        #     communication(1, s2)
            
        torch.cuda.nvtx.range_pop()
        

if __name__=='__main__':
    # warmup
    run()
    start = time.time()
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True, profile_memory=True,
    #     with_stack=True, with_modules=True, with_flops=True,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('profile_try')
    # ) as p:
    #     run()
    run()
    end = time.time()
    print(f"Time: {end - start}")
    torch.distributed.destroy_process_group()