import torch
import torch.distributed as dist
import torch.cuda.amp as amp
import os
import time
import cupy as cp

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
    }
}
''', 'multiply_by_two')

def compute(iter_range_out, iter_range_in, sm_fraction):
    compute_stream = torch.cuda.Stream()
    """Simulated compute kernel with adjustable SM usage."""
    N = 1024 * 1024
    x = torch.ones(N, dtype=torch.float32, device=device)
    threads_per_block = int(256 // sm_fraction)
    num_blocks = (N + threads_per_block - 1) // threads_per_block

    with torch.cuda.stream(compute_stream):
        for _ in range(iter_range_out):
            cuda_kernel((num_blocks,), (threads_per_block,), (x.data_ptr(), N, iter_range_in))
    compute_stream.synchronize()
    torch.cuda.synchronize()

def communication(iter_range):
    comm_stream = torch.cuda.Stream()
    # Allocate tensors
    size = 1024 * 1024 * 1024  # 16M elements (~64MB for fp32)
    tensor_send = torch.ones(size, dtype=torch.float32, device=device)
    tensor_recv = torch.empty(size, dtype=torch.float32, device=device)
    """Conducts P2P communication using full bandwidth."""
    peer = (rank + 1) % world_size  # Simple ring communication
    for _ in range(iter_range):
        with torch.cuda.stream(comm_stream):
            if rank % 2 == 0:
                req_send = dist.isend(tensor_send, peer)
                req_recv = dist.irecv(tensor_recv, peer)
            else:
                req_recv = dist.irecv(tensor_recv, peer)
                req_send = dist.isend(tensor_send, peer)
            req_send.wait()
            req_recv.wait()
        comm_stream.synchronize()
        torch.cuda.synchronize()

if __name__ == "__main__":
    sm_fraction = float(os.getenv("SM_FRACTION", 1.0))  # Adjust computational strength

    dist.barrier(device_ids=[device.index])
    
    start_time = time.time()
    
    # Launch communication and computation in parallel
    communication(10)
    # compute(1000 * 40, 1000, sm_fraction)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    dist.barrier(device_ids=[device.index])  # Ensure all processes finish together
    dist.destroy_process_group()
    
    if rank == 0:
        print(f"Execution time: {end_time - start_time:.4f}s")
