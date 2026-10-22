/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>
// The CUDA Safe runtime does not ship the profiler API; gate it on header availability.
#if defined(__has_include) && __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#define HAVE_CUDA_PROFILER_API 1
#endif

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
// 重点理解 shared memory tiling 的数据复用,
// 以及两次同步分别保护 tile 的加载和下一轮覆盖. 旧笔记关联:
// A, B, C 使用 row-major layout: element(row, col) 的 offset 为 row * width + col.
// 一个 block 负责 C 的一个 BLOCK_SIZE x BLOCK_SIZE tile, 每个 thread 累加一个输出元素.
// kernel 没有边界检查; 输入尺寸必须满足完整 tile 的要求.
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    // 输出坐标: row = by * BLOCK_SIZE + ty, col = bx * BLOCK_SIZE + tx.
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    // 固定 A 的 tile row, 沿 A 的列方向前进; 每次 a 增加 BLOCK_SIZE 个元素.
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    // 固定 B 的 tile column, 沿 B 的行方向前进; 每次 b 增加 BLOCK_SIZE * wB 个元素.
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    // Csub 是每个 thread 私有的 accumulator, 跨所有 tile iteration 累加 dot product.
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    // 每轮计算公共维度 K 中的一个 tile; 默认 K=320, BLOCK_SIZE=32, 共迭代 10 次.
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        // 同一 block 的 threads 共享 As/Bs; 不同 block 拥有独立的 shared memory.
        // BLOCK_SIZE=32 时两个 float tile 合计占 8 KiB, 每轮复用同一份空间.
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        // 每个 thread 从 A/B 各加载一个元素, 全 block 合作填满两个 tile.
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // 第一次 barrier: 等所有 thread 写完 As/Bs, 才允许其他 thread 读取这些值.
        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        // unroll 提示 compiler 展开固定次数的 k loop, 不会创建额外的 thread.
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // 使用 As 的第 ty 行和 Bs 的第 tx 列; tile 内每个输入元素可被多个 thread 复用.
            Csub += As[ty][k] * Bs[k][tx];
        }

        // 第二次 barrier: 等所有 thread 读完当前 tile, 再允许下一轮覆盖 As/Bs.
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    // c 是当前 C tile 左上角的 offset; 加上 thread 的 row/column offset 后写回一个结果.
    int c               = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA, const dim3 &dimsB)
{
    // Allocate host memory for matrices A and B
    // host array 使用 pinned memory, 配合下面同一 stream 内的 async copy.
    unsigned int size_A     = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float       *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A, 0));
    unsigned int size_B     = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float       *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B, 0));
    cudaStream_t stream;

    // Initialize host memory
    // A 全为 1, B 全为 0.01; 每个 C 元素的 reference 都是 K * 0.01.
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    // A 为 M x K, B 为 K x N, 则 C 为 M x N; dim3.x 存 column 数, dim3.y 存 row 数.
    dim3         dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float       *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C, 0));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // H2D, kernel 和 D2H 使用同一个 stream, 依靠 stream order 保证数据依赖.
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    // 默认 block=(32,32), 每个 block 1024 threads; grid=(20,10), 共 200 个 block.
    // 这里用整除而非向上取整, M/N/K 均应为 block_size 的正整数倍.
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // warmup 不计入 event 区间; 后面的 stream synchronize 也等待前面的 H2D 完成.
    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16) {
        MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else {
        MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // start/stop 包围 300 次 kernel launch, 不包含 H2D/D2H; Debug build 的数值不用于性能结论.
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    // GPU 重复执行 300 次, 用总耗时求单次平均值.
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else {
            MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    // 每个输出元素有 K 次 multiply-add, 按 2 FLOPs 计, 整次运算约为 2*M*N*K FLOPs.
    float  msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul =
        2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops,
           msecPerMatrixMul,
           flopsPerMatrixMul,
           threads.x * threads.y);

    // Copy result from device to host
    // D2H 提交后还要等待 stream 完成, CPU 才能安全读取 h_C.
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6; // machine zero

    // 利用常量输入直接计算 reference; 此处 error 还除以 dot_length, 并非普通 relative error.
    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err    = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val    = fabs(h_C[i]);
        double rel_err    = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf("\nNOTE: The CUDA Samples are not meant for performance "
           "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices"
               " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 32;

    // dim3.x 是列数(宽), dim3.y 是行数(高). 默认 block_size=32 时,
    // 按"行 x 列"表示: A 为 320x320, B 为 320x640, C 为 320x640.
    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    // 检查公共维度 K: A 的 column 数必须等于 B 的 row 数.
    // 当前代码没有检查 tile 对齐条件, 使用命令行尺寸时还需自行保证正数且为 32 的倍数.
    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    // 这里输出的是(列数, 行数), 与通常的矩阵(行数, 列数)记法相反.
    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

#ifdef HAVE_CUDA_PROFILER_API
    checkCudaErrors(cudaProfilerStart());
#endif
    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
#ifdef HAVE_CUDA_PROFILER_API
    checkCudaErrors(cudaProfilerStop());
#endif

    exit(matrix_result);
}
