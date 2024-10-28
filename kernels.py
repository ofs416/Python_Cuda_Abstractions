import numpy as np
import torch
import triton
import triton.language as tl
from numba import cuda
import taichi as ti
import cupy as cp
from cupyx import jit
import pycuda.autoinit
import pycuda.driver as cuda_driver
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

# ==============================================
# Numba Implementation
# ==============================================
@cuda.jit
def matmul_numba(A, B, C):
    """Matrix multiplication kernel using Numba."""
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def run_numba(A, B):
    """Runner function for Numba implementation."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    
    threadsperblock = (16, 16)
    blockspergrid_x = (A.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (B.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.to_device(C)
    
    matmul_numba[blockspergrid, threadsperblock](A_device, B_device, C_device)
    
    return C_device.copy_to_host()

# ==============================================
# Taichi Implementation
# ==============================================
ti.init(arch=ti.gpu)

@ti.kernel
def matmul_taichi(A: ti.types.ndarray(), B: ti.types.ndarray(), C: ti.types.ndarray()):
    """Matrix multiplication kernel using Taichi."""
    for i, j in ti.ndrange(A.shape[0], B.shape[1]):
        sum = 0.0
        for k in range(A.shape[1]):
            sum += A[i, k] * B[k, j]
        C[i, j] = sum

def run_taichi(A, B):
    """Runner function for Taichi implementation."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    matmul_taichi(A, B, C)
    return C

# ==============================================
# Triton Implementation
# ==============================================
@triton.jit
def matmul_triton_kernel(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c)

def run_triton(A, B):
    """Runner function for Triton implementation."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    
    M, K = A.shape
    _, N = B.shape
    
    A_gpu = torch.from_numpy(A).cuda()
    B_gpu = torch.from_numpy(B).cuda()
    C_gpu = torch.empty((M, N), device='cuda', dtype=torch.float32)
    
    BLOCK_SIZE = 16
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_triton_kernel[grid](
        A_gpu, B_gpu, C_gpu,
        M, N, K,
        A_gpu.stride(0), A_gpu.stride(1),
        B_gpu.stride(0), B_gpu.stride(1),
        C_gpu.stride(0), C_gpu.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE,
        BLOCK_SIZE_N=BLOCK_SIZE,
        BLOCK_SIZE_K=BLOCK_SIZE,
    )
    
    return C_gpu.cpu().numpy()

# ==============================================
# PyCUDA Implementation
# ==============================================
pycuda_kernel = """
#define BLOCK_SIZE 16

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        if (row < M && tile * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + tile * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (tile * BLOCK_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
"""

class PyCUDAMatMul:
    def __init__(self):
        self.mod = SourceModule(pycuda_kernel)
        self.matmul = self.mod.get_function("matmul")
        self.BLOCK_SIZE = 16
        
    def __call__(self, A, B):
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        
        M, K = A.shape
        _, N = B.shape
        
        C = np.empty((M, N), dtype=np.float32)
        
        grid_x = (N + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        grid_y = (M + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        
        A_gpu = gpuarray.to_gpu(A)
        B_gpu = gpuarray.to_gpu(B)
        C_gpu = gpuarray.empty((M, N), dtype=np.float32)
        
        self.matmul(
            A_gpu, B_gpu, C_gpu,
            np.int32(M), np.int32(N), np.int32(K),
            block=(self.BLOCK_SIZE, self.BLOCK_SIZE, 1),
            grid=(grid_x, grid_y)
        )
        
        return C_gpu.get()

# ==============================================
# CuPy Implementations
# ==============================================
def run_cupy_simple(A, B):
    """Simple CuPy matrix multiplication using @ operator."""
    A = cp.array(A)
    B = cp.array(B)
    return cp.asnumpy(A @ B)

# Raw kernel implementation
cuda_code = r'''
extern "C" __global__
void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int BLOCK_SIZE = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        if (row < M && tile * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + tile * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (tile * BLOCK_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
'''

matmul_kernel_cupy = cp.RawKernel(cuda_code, 'matmul_kernel')

def run_cupy_raw(A, B):
    """Matrix multiplication using raw CUDA kernel."""
    A = cp.array(A)
    B = cp.array(B)
    m, k = A.shape
    _, n = B.shape
    
    C = cp.empty((m, n), dtype=cp.float32)
    
    block_size = 16
    grid_x = (n + block_size - 1) // block_size
    grid_y = (m + block_size - 1) // block_size
    
    matmul_kernel_cupy(
        grid=(grid_x, grid_y),
        block=(block_size, block_size),
        args=(A, B, C, m, n, k)
    )
    
    return cp.asnumpy(C)

# JIT kernel implementation
@jit.rawkernel()
def matmul_jit_kernel(a, b, c, m: int, n: int, k: int):
    """JIT-compiled CUDA kernel for matrix multiplication."""
    BLOCK_SIZE = 16
    
    smem_a = jit.shared_memory(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=float)
    smem_b = jit.shared_memory(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=float)
    
    tx = jit.threadIdx.x
    ty = jit.threadIdx.y
    bx = jit.blockIdx.x
    by = jit.blockIdx.y
    
    row = by * BLOCK_SIZE + ty
    col = bx * BLOCK_SIZE + tx
    
    sum = 0.0
    
    for tile in range((k + BLOCK_SIZE - 1) // BLOCK_SIZE):
        if row < m and tile * BLOCK_SIZE + tx < k:
            smem_a[ty, tx] = a[row, tile * BLOCK_SIZE + tx]
        else:
            smem_a[ty, tx] = 0.0
            
        if tile * BLOCK_SIZE + ty < k and col < n:
            smem_b[ty, tx] = b[tile * BLOCK_SIZE + ty, col]
        else:
            smem_b[ty, tx] = 0.0
            
        jit.sync_threads()
        
        for i in range(BLOCK_SIZE):
            sum += smem_a[ty, i] * smem_b[i, tx]
            
        jit.sync_threads()
    
    if row < m and col < n:
        c[row, col] = sum

def run_cupy_jit(A, B):
    """Matrix multiplication using JIT-compiled kernel."""
    A = cp.array(A)
    B = cp.array(B)
    m, k = A.shape
    _, n = B.shape
    
    C = cp.empty((m, n), dtype=cp.float32)
    
    block_size = 16
    grid_x = (n + block_size - 1) // block_size
    grid_y = (m + block_size - 1) // block_size
    
    matmul_jit_kernel(
        (grid_x, grid_y),
        (block_size, block_size),
        (A, B, C, m, n, k)
    )
    
    return cp.asnumpy(C)
