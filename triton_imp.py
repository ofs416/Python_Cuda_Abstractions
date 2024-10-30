import numpy as np
import torch
import triton
import triton.language as tl

# Triton implementation
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    width = grid_n
    group_m = pid // width
    group_n = pid % width
    
    offs_am = (group_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (group_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # Compute matmul
        accumulator += tl.dot(a, b)
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    offs_cm = group_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = group_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)


if __name__ == "__main__":
    # Set device
    torch.cuda.set_device(0)
    
    size = 4096

    x_h = np.random.rand(size, size).astype(np.float32)
    y_h = np.random.rand(size, size).astype(np.float32)
    z_h = np.zeros([size, size], dtype=np.float32)

    x_d = torch.from_numpy(x_h).cuda()
    y_d = torch.from_numpy(y_h).cuda()
    z_d = torch.zeros((size, size), dtype=torch.float32, device='cuda')

    grid = lambda META: (
        triton.cdiv(size, META['BLOCK_SIZE_M']) * triton.cdiv(size, META['BLOCK_SIZE_N']),
    )

    # Warmup
    print("Warming up...")
    for _ in range(10):
        matmul_kernel[grid](
            a_ptr=x_d, b_ptr=y_d, c_ptr=z_d,
            M=size, N=size, K=size,
            stride_am=size, stride_ak=1,
            stride_bk=size, stride_bn=1,
            stride_cm=size, stride_cn=1,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        )
        torch.matmul(x_d, y_d)
    torch.cuda.synchronize()

    # Benchmark Triton
    print("\nBenchmarking Triton...")
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    for _ in range(20):
        matmul_kernel[grid](
            a_ptr=x_d, b_ptr=y_d, c_ptr=z_d,
            M=size, N=size, K=size,
            stride_am=size, stride_ak=1,
            stride_bk=size, stride_bn=1,
            stride_cm=size, stride_cn=1,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        )
    
    end_event.record()
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event) / 20
    print(f"Triton execution time: {triton_time:.2f} ms")

    # Benchmark PyTorch
    print("\nBenchmarking PyTorch...")
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    for _ in range(20):
        torch.matmul(x_d, y_d)
    
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / 20
    print(f"PyTorch execution time: {torch_time:.2f} ms")