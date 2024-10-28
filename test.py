import numpy as np
import torch
import triton
import triton.language as tl
import taichi as ti
import numba
from numba import cuda
import time

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
    
    # Program ID -> (m, n) block
    width = grid_n
    group_m = pid // width
    group_n = pid % width
    
    # Initialize pointers to A, B, C
    offs_am = (group_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (group_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # Compute matmul
        accumulator += tl.dot(a, b)
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store result
    offs_cm = group_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = group_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)

# Taichi implementation
ti.init(arch=ti.gpu)

BLOCK_SIZE = 32

@ti.kernel
def matmul_taichi(
    A: ti.template(), 
    B: ti.template(), 
    C: ti.template()
):
    # Configure block-level parallelism
    ti.loop_config(block_dim=BLOCK_SIZE)
    for i, j in ti.ndrange(A.shape[0], B.shape[1]):
        # Use local accumulator for better performance
        sum = 0.0
        # Configure vectorization for inner loop
        for k in range(A.shape[1]):
            sum += A[i, k] * B[k, j]
        C[i, j] = sum

# Numba implementation
BLOCK_DIM = 32  # Must be power of 2

@cuda.jit
def matmul_numba(A, B, C):
    # Shared memory arrays for the sub-matrices
    tile_A = cuda.shared.array(shape=(BLOCK_DIM, BLOCK_DIM), dtype=np.float32)
    tile_B = cuda.shared.array(shape=(BLOCK_DIM, BLOCK_DIM), dtype=np.float32)
    
    row, col = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    # Each thread computes one element in the result matrix
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        # Loop over tiles
        for p in range((A.shape[1] + BLOCK_DIM - 1) // BLOCK_DIM):
            # Collaboratively load tiles into shared memory
            if p * BLOCK_DIM + ty < A.shape[1] and row < A.shape[0]:
                tile_A[ty, tx] = A[row, p * BLOCK_DIM + ty]
            else:
                tile_A[ty, tx] = 0.0
                
            if p * BLOCK_DIM + ty < B.shape[0] and col < B.shape[1]:
                tile_B[ty, tx] = B[p * BLOCK_DIM + ty, col]
            else:
                tile_B[ty, tx] = 0.0
            
            # Wait until all threads load their data
            cuda.syncthreads()
            
            # Compute partial dot product
            for i in range(BLOCK_DIM):
                tmp += tile_A[tx, i] * tile_B[i, ty]
            
            # Wait until all threads are done with the tile
            cuda.syncthreads()
        
        C[row, col] = tmp

def benchmark(size=1024):
    # Generate random matrices
    a_np = np.random.randn(size, size).astype(np.float32)
    b_np = np.random.randn(size, size).astype(np.float32)
    
    # Convert to PyTorch tensors for Triton and PyTorch
    a = torch.from_numpy(a_np).cuda()
    b = torch.from_numpy(b_np).cuda()
    c_triton = torch.zeros((size, size), dtype=torch.float32, device='cuda')
    
    # Triton benchmark
    grid = lambda META: (
        triton.cdiv(size, META['BLOCK_SIZE_M']) * triton.cdiv(size, META['BLOCK_SIZE_N']),
    )
    
    # Warmup Triton
    for _ in range(10):
        matmul_kernel[grid](
            a_ptr=a, b_ptr=b, c_ptr=c_triton,
            M=size, N=size, K=size,
            stride_am=size, stride_ak=1,
            stride_bk=size, stride_bn=1,
            stride_cm=size, stride_cn=1,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        )
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c_triton,
        M=size, N=size, K=size,
        stride_am=size, stride_ak=1,
        stride_bk=size, stride_bn=1,
        stride_cm=size, stride_cn=1,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
    )
    torch.cuda.synchronize()
    triton_time = time.perf_counter() - start
    
    # PyTorch GPU benchmark
    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    c_pytorch = torch.matmul(a, b)
    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start
    
    # Taichi benchmark
    a_ti = ti.field(float, shape=(size, size))
    b_ti = ti.field(float, shape=(size, size))
    a_ti.from_numpy(a_np)
    b_ti.from_numpy(b_np)

    c_taichi = ti.field(float, shape=(size, size))
    # Warmup
    for _ in range(10):
        matmul_taichi(a_ti, b_ti, c_taichi)
        ti.sync()
    
    start = time.perf_counter()
    matmul_taichi(a_ti, a_ti, c_taichi)
    ti.sync()
    taichi_time = time.perf_counter() - start
    
    # Numba benchmark
    c_numba = np.zeros((size, size), dtype=np.float32)
    threadsperblock = (16, 16)
    blockspergrid_x = (size + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (size + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    a_cuda = cuda.to_device(a_np)
    b_cuda = cuda.to_device(b_np)
    c_cuda = cuda.to_device(c_numba)
    
    # Warmup
    for _ in range(10):
        matmul_numba[blockspergrid, threadsperblock](a_cuda, b_cuda, c_cuda)
        cuda.synchronize()
    
    start = time.perf_counter()
    matmul_numba[blockspergrid, threadsperblock](a_cuda, b_cuda, c_cuda)
    cuda.synchronize()
    numba_time = time.perf_counter() - start
    
    # NumPy benchmark (CPU, for reference)
    start = time.perf_counter()
    c_numpy = np.matmul(a_np, b_np)
    numpy_time = time.perf_counter() - start
    
    # Get results back to CPU for verification
    c_triton_np = c_triton.cpu().numpy()
    c_pytorch_np = c_pytorch.cpu().numpy()
    c_numba_np = c_cuda.copy_to_host()
    
    # Calculate TFLOPS (Tera Floating Point Operations per Second)
    # Matrix multiplication requires 2*M*N*K operations
    ops = 2 * size * size * size  # Multiply-adds count as 2 operations
    def get_tflops(time_seconds):
        return (ops / 1e12) / time_seconds
    
    # Print results
    print(f"\nMatrix size: {size}x{size}")
    print(f"{'Framework':<10} {'Time (ms)':<12} {'TFLOPS':<8}")
    print("-" * 30)
    print(f"{'Triton':<10} {triton_time*1000:>9.2f}ms {get_tflops(triton_time):>8.2f}")
    print(f"{'PyTorch':<10} {pytorch_time*1000:>9.2f}ms {get_tflops(pytorch_time):>8.2f}")
    print(f"{'Taichi':<10} {taichi_time*1000:>9.2f}ms {get_tflops(taichi_time):>8.2f}")
    print(f"{'Numba':<10} {numba_time*1000:>9.2f}ms {get_tflops(numba_time):>8.2f}")
    print(f"{'NumPy':<10} {numpy_time*1000:>9.2f}ms {get_tflops(numpy_time):>8.2f}")
    

if __name__ == "__main__":
    np.random.seed(42)
    benchmark(8192) 