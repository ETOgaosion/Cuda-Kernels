import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
import time

# Initialize distributed environment
print(f"Rank {os.getenv('RANK')} in {os.getenv('WORLD_SIZE')} with {os.getenv('MASTER_ADDR')}:{os.getenv('MASTER_PORT')} initializing...")
dist.init_process_group(backend='nccl', init_method=f"tcp://{os.getenv('MASTER_ADDR')}:{os.getenv('MASTER_PORT')}", rank=int(os.getenv('RANK')), world_size=int(os.getenv('WORLD_SIZE')))
print(f"Rank {dist.get_rank()} initialized.")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

class SimpleModel(nn.Module):
    def __init__(self, strength=1024, layers=24):
        super(SimpleModel, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(strength, strength)] * layers)

    def forward(self, x):
        for layer in self.fc:
            x = layer(x)
        return x

def get_model_and_optimizer(rank, strength=1024, layers=24):
    model = SimpleModel(strength=strength, layers=layers).to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    return model, optimizer

def communication(iter_range, comm_stream):
    # Allocate tensors
    size = 1024 * 1024 * 1024  # 16M elements (~64MB for fp32)
    tensor_send = torch.ones(size, dtype=torch.float32, device=torch.cuda.current_device)
    tensor_recv = torch.empty(size, dtype=torch.float32, device=torch.cuda.current_device)
    """Conducts P2P communication using full bandwidth."""
    rank = torch.distributed.get_rank()
    peer = (rank + 1) % world_size  # Simple ring communication
    reqs = []  # 保存所有通信请求

    with torch.cuda.stream(comm_stream):
        for _ in range(iter_range):
            send_op = dist.P2POp(dist.isend, tensor_send, (rank + 1)%world_size)
            recv_op = dist.P2POp(dist.irecv, tensor_recv, (rank + 1)%world_size)
            reqs = dist.batch_isend_irecv([send_op, recv_op])

    for req in reqs:
        req.wait()

def train():
    # 创建CUDA流
    compute_stream = torch.cuda.Stream(device=rank)
    comm_stream = torch.cuda.Stream(device=rank)

    # 模拟输入数据
    strength = 1024
    layers = 24
    model, optimizer = get_model_and_optimizer(rank, strength=strength, layers=layers)
    input_data = torch.randn(strength, strength, device=rank)
    target = torch.randn(strength, strength, device=rank)

    start = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        record_shapes=True, profile_memory=True,
        with_stack=True, with_modules=True, with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('profile_model')
    ) as p:
        for _ in range(3):  # 模拟10个训练步骤
            optimizer.zero_grad()

            # 在计算流中进行前向传播
            with torch.cuda.stream(compute_stream):
                output = model(input_data)
                loss = nn.functional.mse_loss(output, target)

            # 在通信流中进行梯度同步
            with torch.cuda.stream(comm_stream):
                loss.backward()
                for param in model.parameters():
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)

            # 等待计算流和通信流完成
            torch.cuda.synchronize(rank)

            # 更新模型参数
            optimizer.step()
            
            p.step()
    end = time.time()
    print(f"Time: {end-start}")

train()