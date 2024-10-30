import numpy as np
from numba import jit, prange
import time

@jit(nopython=True, parallel=True)
def matrix_multiply_parallel(A, B):

    m, k = A.shape
    k, n = B.shape
    result = np.zeros((m, n))
    
    # Parallelize the outer loop
    for i in prange(m):
        for j in range(n):
            tmp = 0.0
            # Inner loop for dot product
            for p in range(k):
                tmp += A[i, p] * B[p, j]
            result[i, j] = tmp
        

if __name__ == "__main__":

    size = 256
    
    A = np.random.random((size, size))
    B = np.random.random((size, size))
    
    # Warm up the JIT compiler
    small_A = np.random.random((10, 10))
    small_B = np.random.random((10, 10))

    matrix_multiply_parallel(A, B)
        
    
    # Time our implementation
    start = time.time()
    for _ in range(1):
        matrix_multiply_parallel(A, B)
    parallel_time = (time.time() - start)
    
    # Time numpy's implementation
    start = time.time()
    for _ in range(1):
        np.matmul(A, B)
    numpy_time = (time.time() - start)

    
    print(f"Numba CPU execution time: {parallel_time} ms")
    print(f"Numpy CPU execution time: {numpy_time} ms")
    