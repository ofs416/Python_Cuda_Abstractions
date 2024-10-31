import cupy as cp
import numpy as np
import torch
import math

# Thread block size
TPB = 16

# CUDA kernel definition
cuda_code = r'''
extern "C" __global__ void fast_matmul(const float* A, const float* B, float* C, 
                                     int M, int N, int K) {
    // Block row and column
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread row and column within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column that this thread computes
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // Shared memory for the sub-matrices
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];
    
    float tmp = 0.0f;
    
    // Loop over the sub-matrices
    for (int m = 0; m < (K + 15) / 16; ++m) {
        // Load the matrices from global memory to shared memory
        if (row < M && m * 16 + tx < K) {
            sA[ty][tx] = A[row * K + m * 16 + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }
        
        if (col < N && m * 16 + ty < K) {
            sB[ty][tx] = B[(m * 16 + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure the sub-matrices are loaded
        __syncthreads();
        
        // Multiply the sub-matrices
        for (int k = 0; k < 16; ++k) {
            tmp += sA[ty][k] * sB[k][tx];
        }
        
        // Synchronize to ensure that the preceding computation 
        // is done before loading new sub-matrices
        __syncthreads();
    }
    
    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = tmp;
    }
}
'''

if __name__ == "__main__":
    # Compile the CUDA kernel
    module = cp.RawModule(code=cuda_code)
    kernel = module.get_function('fast_matmul')
    
    size = 4096
    
    # Generate random matrices
    x_h = np.random.rand(size, size).astype(np.float32)
    y_h = np.random.rand(size, size).astype(np.float32)
    
    # Transfer to GPU
    x_d = cp.asarray(x_h)
    y_d = cp.asarray(y_h)
    z_d = cp.zeros((size, size), dtype=cp.float32)
    
    # Calculate grid dimensions
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(size / threadsperblock[0])
    blockspergrid_y = math.ceil(size / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Warm-up runs
    for _ in range(10):
        kernel(
            block=threadsperblock,
            grid=blockspergrid,
            args=(x_d, y_d, z_d, size, size, size)
        )
        cp.cuda.stream.get_current_stream().synchronize()
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Benchmark runs
    start_event.record()
    for _ in range(20):
        kernel(
            block=threadsperblock,
            grid=blockspergrid,
            args=(x_d, y_d, z_d, size, size, size)
        )
        cp.cuda.stream.get_current_stream().synchronize()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 20  # Average time in milliseconds
    
    print(f"CuPy custom kernel execution time: {elapsed_time:.2f} ms")
