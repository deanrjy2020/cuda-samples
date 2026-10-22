/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 *
 * 示例流程: 分配统一内存 -> CPU 初始化 -> GPU 并行相加 -> 同步等待
 * -> CPU 串行计算并校验 -> 释放内存.
 * 注意: 当前实现未检查 CUDA API 的返回值, 使用时需确保调用成功.
 */

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements. This exmple shows Vector addition using Unified memory.
 */


// __global__ 表示由 CPU 发起调用, 在 GPU 上执行的核函数; 每个线程处理一个元素.
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    if (blockIdx.x == 2 && threadIdx.x == 3)
    {
        printf("Debug: block=%u thread=%u\n", blockIdx.x, threadIdx.x);
    }
    // 一维全局索引 = 块内线程编号 + 线程块编号 * 每块线程数.
    // host指定的是4*256, B和T都是一维的, 只考虑x就可以.
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    // 线程总数可能大于向量长度, 最后一个块中多余的线程必须跳过访问.
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

// 在 CPU 上用 [0, 1] 范围内的伪随机数初始化数组.
void initArray(float* A, int length)
{
    // 每次调用都重新设置种子; 同一秒内初始化 A, B 时可能产生相同序列.
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

// CPU 串行参考实现, 用于验证 GPU 的计算结果.
void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// 逐元素比较绝对误差, 允许浮点运算存在 epsilon 范围内的差异.
// 这里用于本示例的有限数值, 不专门处理 NaN 等特殊浮点值.
bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    // 默认计算 1024 个元素; 第一个命令行参数可覆盖长度.
    // 当前没有输入校验, 调用者应传入有效的正整数.
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }

    //unified-memory-example-begin

    // Pointers to memory vectors
    // A, B 为输入, C 为 GPU 输出; comparisonResult 保存 CPU 参考结果.
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    // malloc 仅分配主机内存; 分配大小按"元素个数 * 每个元素的字节数"计算.
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    // 分配 CPU 和 GPU 均可访问的统一内存, CUDA 负责管理数据的可访问性及迁移.
    // 本示例因此不需要显式调用 cudaMemcpy 来传输输入和输出.
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    // CPU 直接写入统一内存中的输入数组.
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    // 每个线程块包含 256 个线程, 向上取整计算块数, 确保覆盖全部元素.
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    // <<<线程块数, 每块线程数>>> 指定启动配置; 核函数启动相对 CPU 是异步的.
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    // 等待 GPU 完成计算, 之后 CPU 才能安全地访问这些统一内存数据并校验结果.
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    // 使用相同输入在 CPU 上串行计算, 生成参考结果.
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    // 对比 GPU 输出和 CPU 参考结果, 并打印校验结果.
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    // 统一内存由 cudaFree 释放; malloc 分配的主机内存由 free 释放.
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

    //unified-memory-example-end	

    return 0;
}
