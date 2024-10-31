import cupy as cp
import numpy as np
import torch
import time

if __name__ == "__main__":
    # Ensure CuPy is using the same device as PyTorch
    cp.cuda.runtime.setDevice(torch.cuda.current_device())
    
    size = 4096

    # Generate random matrices on CPU
    x_h = np.random.rand(size, size).astype(np.float32)
    y_h = np.random.rand(size, size).astype(np.float32)

    # Transfer to GPU
    x_d = cp.asarray(x_h)
    y_d = cp.asarray(y_h)
    
    # Warm-up runs
    for _ in range(10):
        z_d = cp.matmul(x_d, y_d)
        cp.cuda.stream.get_current_stream().synchronize()

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark runs
    start_event.record()
    for _ in range(20):
        z_d = cp.matmul(x_d, y_d)
        cp.cuda.stream.get_current_stream().synchronize()
    end_event.record()

    torch.cuda.synchronize()  # Wait for all operations to finish
    elapsed_time = start_event.elapsed_time(end_event) / 20  # Average time in milliseconds
 
    print(f"CuPy execution time: {elapsed_time:.2f} ms")
