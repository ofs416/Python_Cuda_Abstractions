import numpy as np
import torch
import time
import pandas as pd
import os

# Fix for X11 error
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['DISPLAY'] = ':0'

def run_benchmark(func, A, B, name="", num_runs=10, warmup_runs=3):
    """Generic benchmark runner for any implementation."""
    # Ensure inputs are float32
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    
    try:
        # Warmup runs
        for _ in range(warmup_runs):
            result = func(A, B)
        
        # Actual timing runs
        times = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            result = func(A, B)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate GFLOPS
        m, k = A.shape
        _, n = B.shape
        flops = 2 * m * n * k  # multiply-add is 2 operations
        gflops = flops / (mean_time * 1e-3) / 1e9
        
        return {
            'name': name,
            'mean_ms': mean_time,
            'std_ms': std_time,
            'gflops': gflops,
            'times': times,
            'result': result
        }
    except Exception as e:
        print(f"{name:<20} Failed: {str(e)}")
        return None

def numpy_matmul(A, B):
    return np.dot(A, B)

def torch_matmul(A, B):
    A_torch = torch.from_numpy(A).cuda()
    B_torch = torch.from_numpy(B).cuda()
    C_torch = torch.matmul(A_torch, B_torch)
    return C_torch.cpu().numpy()

def compare_all_implementations(M=1024, N=1024, K=1024, num_runs=10, warmup_runs=3):
    """Compare all implementations with the same input matrices."""
    # Generate random matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Start with basic implementations
    implementations = [
        ('NumPy (CPU)', numpy_matmul),
        ('PyTorch (cuBLAS)', torch_matmul),
    ]
    
    # Try to import and add other implementations
    try:
        from kernels import run_numba
        implementations.append(('Numba', run_numba))
    except ImportError as e:
        print(f"Numba import failed: {e}")
        
    try:
        from kernels import run_taichi
        implementations.append(('Taichi', run_taichi))
    except ImportError as e:
        print(f"Taichi import failed: {e}")
        
    try:
        from kernels import run_triton
        implementations.append(('Triton', run_triton))
    except ImportError as e:
        print(f"Triton import failed: {e}")
        
    try:
        from kernels import PyCUDAMatMul
        implementations.append(('PyCUDA', lambda a, b: PyCUDAMatMul()(a, b)))
    except ImportError as e:
        print(f"PyCUDA import failed: {e}")
        
    try:
        import cupy as cp
        from kernels import run_cupy_simple, run_cupy_raw, run_cupy_jit
        implementations.extend([
            ('CuPy Simple', run_cupy_simple),
            ('CuPy Raw', run_cupy_raw),
            ('CuPy JIT', run_cupy_jit)
        ])
    except ImportError as e:
        print(f"CuPy import failed: {e}")
    
    # Run benchmarks
    results = []
    reference = None
    
    print(f"\nBenchmarking matrix multiplication (M={M}, N={N}, K={K})")
    print("-" * 80)
    print(f"{'Implementation':<20} {'Time (ms)':<15} {'Std (ms)':<15} {'GFLOPS':<10} {'Max Error':<10}")
    print("-" * 80)
    
    for name, impl in implementations:
        result = run_benchmark(impl, A, B, name, num_runs, warmup_runs)
        if result is not None:
            # Use PyTorch result as reference for error checking
            if name == 'PyTorch (cuBLAS)':
                reference = result['result']
            
            # Calculate max error if reference is available
            max_error = np.max(np.abs(result['result'] - reference)) if reference is not None else 0
            
            print(f"{name:<20} {result['mean_ms']:>8.2f} ±{result['std_ms']:>6.2f} "
                  f"{result['gflops']:>10.1f} {max_error:>10.2e}")
            
            results.append({
                'name': name,
                'mean_ms': result['mean_ms'],
                'std_ms': result['std_ms'],
                'gflops': result['gflops'],
                'max_error': max_error
            })
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=['name', 'mean_ms', 'std_ms', 'gflops', 'max_error'])

if __name__ == "__main__":
    # Test different matrix sizes
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192)
    ]
    
    for M, N, K in sizes:
        print(f"\nTesting size: {M}x{K} @ {K}x{N}")
        results_df = compare_all_implementations(M, N, K)
        
        # Save results to CSV
        filename = f"matmul_benchmark_results_{M}x{N}.csv"
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")