import taichi as ti
import numpy as np

# Initialize Taichi with GPU backend
ti.init(kernel_profiler=True, arch=ti.gpu)

@ti.kernel
def matmul(
    M: ti.i32,
    N: ti.i32,
    K: ti.i32
):
    ti.loop_config(block_dim=32)
    for i, j in ti.ndrange(M, N):
        sum = 0.0
        for k in range(K):
            sum += x[i, k] * y[k, j]
        z[i, j] = sum


# Example usage
if __name__ == "__main__":
    size = 4096

    x_np = np.random.rand(size, size).astype(np.float32)
    y_np = np.random.rand(size, size).astype(np.float32)
    z_np = np.zeros([size, size], dtype=np.float32)

    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    z = ti.field(ti.f32)
    ti.root.dense(ti.ij, (size, size)).place(x)
    ti.root.dense(ti.ij, (size, size)).place(y)
    ti.root.dense(ti.ij, (size, size)).place(z)


    for _ in range(10):
        matmul(size, size, size)

    for _ in range(20):
        matmul(size, size, size)
    ti.profiler.print_kernel_profiler_info() 