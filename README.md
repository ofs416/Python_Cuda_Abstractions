# Python_Cuda_Abstractions

This will help me gauge an approximation of the performance and ease of use trade-off offered by differing levels of abstractions. Past projects are considerably slow due to the amount of linear algebra computations required; some of these projects currently rely on numpy, so a Library such as CuPy could be heavily beneficial. Others rely on custom functions, so Triton or Taichi may be beneficial for utilising the GPU.

## Running

As I don't own a cuda capable GPU, I launch a cloud instance with a docker container. After using SSH through VS code I run the following commands:

1. source /opt/conda/etc/profile.d/conda.sh
2. conda activate base  # or whichever environment you want to use
3. apt-get update
4. apt-get install -y libx11-6 libxcursor1 libxinerama1 libxrandr2 libxext6
