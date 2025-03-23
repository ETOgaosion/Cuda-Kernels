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
void multiply_by_two(float* data, int N, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < range; i++){
        if (idx < N) {
            data[idx] *= 2;
        }
        __nanosleep(10);
    }
}
''', 'multiply_by_two')

compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

def compute(iter_range, sm_fraction):
    """Simulated compute kernel with adjustable SM usage."""
    N = 1024 * 1024
    x = torch.ones(N, dtype=torch.float32, device=device)
    threads_per_block = int(256 * sm_fraction)
    num_blocks = (N + threads_per_block - 1) // threads_per_block

    with torch.cuda.stream(compute_stream):
        cuda_kernel((num_blocks,), (threads_per_block,), (x.data_ptr(), N, iter_range))

def communication(iter_range):
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
        compute(1000 * 1000 * 40, sm_fraction)
        # communication(10)
    
        torch.cuda.current_stream().wait_stream(compute_stream)
        # torch.cuda.current_stream().wait_stream(comm_stream)
        torch.cuda.synchronize()
    
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
