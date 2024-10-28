import numpy as np
import torch
import cupy as cp
import time
import pycuda.driver as cuda

def run_benchmark(func, A, B, name="", num_runs=10, warmup_runs=3):
    """Generic benchmark runner for any implementation."""
    # Ensure inputs are float32
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    
    # Warmup runs
    for _ in range(warmup_runs):
        result = func(A, B)
    
    # Actual timing runs
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()  # Ensure GPU is synchronized
        start = time.perf_counter()
        
        result = func(A, B)
        
        torch.cuda.synchronize()  # Ensure GPU is synchronized
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
    
    # Return results
    return {
        'name': name,
        'mean_ms': mean_time,
        'std_ms': std_time,
        'gflops': gflops,
        'times': times,
        'result': result
    }

def run_numba(A, B):
    """Run Numba implementation."""
    from kernels import run_numba as numba_impl
    return numba_impl(A, B)

def run_taichi(A, B):
    """Run Taichi implementation."""
    from kernels import run_taichi as taichi_impl
    return taichi_impl(A, B)

def run_triton(A, B):
    """Run Triton implementation."""
    from kernels import run_triton as triton_impl
    return triton_impl(A, B)

def run_pycuda(A, B):
    """Run PyCUDA implementation."""
    from kernels import PyCUDAMatMul
    matmul = PyCUDAMatMul()
    return matmul(A, B)

def run_cupy_simple(A, B):
    """Run simple CuPy implementation using @ operator."""
    from kernels import run_cupy_simple as cupy_simple_impl
    return cupy_simple_impl(A, B)

def run_cupy_raw(A, B):
    """Run CuPy implementation with raw kernel."""
    from kernels import run_cupy_raw as cupy_raw_impl
    return cupy_raw_impl(A, B)

def run_cupy_jit(A, B):
    """Run CuPy implementation with JIT kernel."""
    from kernels import run_cupy_jit as cupy_jit_impl
    return cupy_jit_impl(A, B)

def run_torch(A, B):
    """Run PyTorch implementation for baseline comparison."""
    A_torch = torch.from_numpy(A).cuda()
    B_torch = torch.from_numpy(B).cuda()
    C_torch = torch.matmul(A_torch, B_torch)
    return C_torch.cpu().numpy()

def compare_all_implementations(M=1024, N=1024, K=1024, num_runs=10, warmup_runs=3):
    """Compare all implementations with the same input matrices."""
    # Generate random matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Define all implementations to test
    implementations = [
        ('NumPy (CPU)', lambda a, b: np.dot(a, b)),
        ('PyTorch (cuBLAS)', run_torch),
        ('Numba', run_numba),
        ('Taichi', run_taichi),
        ('Triton', run_triton),
        ('PyCUDA', run_pycuda),
        ('CuPy Simple', run_cupy_simple),
        ('CuPy Raw', run_cupy_raw),
        ('CuPy JIT', run_cupy_jit)
    ]
    
    # Run benchmarks
    results = []
    reference = None
    
    print(f"\nBenchmarking matrix multiplication (M={M}, N={N}, K={K})")
    print("-" * 80)
    print(f"{'Implementation':<20} {'Time (ms)':<15} {'Std (ms)':<15} {'GFLOPS':<10} {'Max Error':<10}")
    print("-" * 80)
    
    for name, impl in implementations:
        try:
            result = run_benchmark(impl, A, B, name, num_runs, warmup_runs)
            
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
            
        except Exception as e:
            print(f"{name:<20} Failed: {str(e)}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test different matrix sizes
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048)
    ]
    
    for M, N, K in sizes:
        print(f"\nTesting size: {M}x{K} @ {K}x{N}")
        results_df = compare_all_implementations(M, N, K)
        
        # Save results to CSV
        filename = f"matmul_benchmark_results_{M}x{N}.csv"
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
