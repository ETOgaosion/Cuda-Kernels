import torch
import torch.distributed as dist
import torch.cuda.amp as amp
import os
import time
import cupy as cp
from threading import Thread

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

compute_stream = torch.cuda.Stream(device=device)
comm_stream = torch.cuda.Stream(device=device)

def compute(iter_range, sm_fraction):
    """Simulated compute kernel with adjustable SM usage."""
    N = 1024 * 8
    x = torch.rand((N, N), dtype=torch.float32, device=device)
    y = torch.rand((N, N), dtype=torch.float32, device=device)
    z = torch.empty((N, N), dtype=torch.float32, device=device)
    threads_per_block = int(256 * sm_fraction)
    num_blocks = (N + threads_per_block - 1) // threads_per_block

    with torch.cuda.stream(compute_stream):
        cuda_kernel((num_blocks,), (threads_per_block,), (x.data_ptr(), y.data_ptr(), z.data_ptr(), N, N, N, iter_range))
        # z = x.matmul(y)

def pre_communication():
    # Allocate tensors
    size = 1024 * 1024 * 1024  # 16M elements (~64MB for fp32)
    tensor_send = torch.ones(size, dtype=torch.float32, device=device)
    tensor_recv = torch.empty(size, dtype=torch.float32, device=device)
    return tensor_send, tensor_recv

def communication(iter_range, tensor_send, tensor_recv):
    """Conducts P2P communication using full bandwidth."""
    peer = (rank + 1) % world_size  # Simple ring communication
    reqs = []  # 保存所有通信请求

    with torch.cuda.stream(comm_stream):
        for _ in range(iter_range):
            # send_op = dist.P2POp(dist.isend, tensor_send, (rank + 1)%world_size)
            # recv_op = dist.P2POp(dist.irecv, tensor_recv, (rank + 1)%world_size)
            # reqs = dist.batch_isend_irecv([send_op, recv_op])
            if rank % 2 == 0:
                reqs.append(dist.isend(tensor_send, peer))
                reqs.append(dist.irecv(tensor_recv, peer))
            else:
                reqs.append(dist.irecv(tensor_recv, peer))
                reqs.append(dist.isend(tensor_send, peer))

    for req in reqs:
        req.wait()

if __name__ == "__main__":
    sm_fraction = float(os.getenv("SM_FRACTION", 0.5))  # Adjust computational strength

    dist.barrier(device_ids=[device.index])
    
    start_time = time.time()
    
    # Launch communication and computation in parallel
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True,
        with_stack=True, with_modules=True, with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('profile')
    ) as p:
        for _ in range(10):
            tensor_send, tensor_recv = pre_communication()
            compute(1000 * 2, sm_fraction)
            communication(1, tensor_recv=tensor_recv, tensor_send=tensor_send)
    
        # torch.cuda.current_stream().wait_stream(compute_stream)
        # torch.cuda.current_stream().wait_stream(comm_stream)
        # torch.cuda.synchronize()
    
        # t1 = Thread(target=compute, args=(1000 * 1000 * 40, sm_fraction))
        # t2 = Thread(target=communication, args=(10,))
        # t1.start()
        # t2.start()
        # t1.join()
        # t2.join()
    
    # compute(1000 * 1000 * 40, sm_fraction)
    # communication(10)

    # torch.cuda.current_stream().wait_stream(compute_stream)
    # torch.cuda.current_stream().wait_stream(comm_stream)
    # torch.cuda.synchronize()
    end_time = time.time()
    
    dist.barrier(device_ids=[device.index])  # Ensure all processes finish together
    dist.destroy_process_group()
    
    if rank == 0:
        print(f"Execution time: {end_time - start_time:.4f}s")
