from numba import cuda, float32
import numpy as np
import math

import torch

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
# TPB should not be larger than 32 in this example
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B using CUDA shared memory.

    Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella
    """
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx + i * TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty + i * TPB) < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp


if __name__ == "__main__":

    size = 4096

    x_h = np.random.rand(size, size).astype(np.float32)
    y_h = np.random.rand(size, size).astype(np.float32)
    z_h = np.zeros([size, size], dtype=np.float32)

    x_d = cuda.to_device(x_h)
    y_d = cuda.to_device(y_h)
    z_d = cuda.to_device(z_h)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    for _ in range(10):
        fast_matmul[blockspergrid, threadsperblock](x_d, y_d, z_d)
        cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(20):
        fast_matmul[blockspergrid, threadsperblock](x_d, y_d, z_d)
        cuda.synchronize()
    end_event.record()

    torch.cuda.synchronize()  # Wait for the kernel to finish
    elapsed_time = start_event.elapsed_time(end_event) / 20 # Time in milliseconds
    print(f"Numba execution time: {elapsed_time} ms")

 