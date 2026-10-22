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

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// Each thread calls device-side printf, printing its linear block index, its
// linear thread index, and val. Output is flushed on host sync; line order is
// not deterministic since blocks/threads run concurrently.
// 第一个值是展平的 block ID [0,3],
// 第二个值是 block 内展平的 thread ID [0,7](对应 main 中的默认配置).
// __global__ 核函数由 CPU 启动, 在 GPU 上执行; 每个线程都会执行一次函数体.
__global__ void simplePrintfKernel(int val)
{
    // 将多维坐标展平成编号时, x 变化最快, 其次是 y, 最后是 z.
    // 本例 grid 的 z 为 1, 所以 block 编号只需计算 y * gridDim.x + x.
    // thread 编号属于各自的 block, 不是整个 grid 中唯一的全局线程编号.
    printf("[%d, %d]:\t\tValue is:%d\n",
           blockIdx.y * gridDim.x + blockIdx.x,                                            // linear block index
           threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x, // linear thread index
           val);
}

int main(int argc, char **argv)
{
    // Select device 0 as the active GPU
    int devID = 0;
    // 为当前主机线程选择后续 CUDA 操作使用的设备, 编号从 0 开始.
    cudaSetDevice(devID);

    // Query compute capability (major.minor) and number of SMs on the device
    // major.minor 表示 GPU 的计算能力版本; SM 是执行线程块的流式多处理器.
    // SM 数量不等于本次启动的 block 数量, 线程块由 GPU 调度到 SM 上执行.
    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devID);

    // Print device info
    printf("GPU Device %d: with compute capability %d.%d and Number of SMs %d\n\n", devID, major, minor, smCount);

    printf("printf() is called. Output:\n\n");

    // Kernel configuration, where a two-dimensional grid and
    // three-dimensional blocks are configured.
    // 2x2=4 个 blocks, 每个 block 有 2x2x2=8 个 threads; 打印顺序不固定.
    // dim3 未指定的维度默认为 1, 因此这里 grid 的实际尺寸是 (2, 2, 1).
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    // <<<block个数, thread个数>>> 启动 32 个线程, 每个线程收到的 val 都是 10.
    // 核函数相对 CPU 异步执行; 设备端 printf 使用缓冲区, 输出顺序不保证按编号排列.
    simplePrintfKernel<<<dimGrid, dimBlock>>>(10);
    // 等待设备计算完成, 并刷新设备端 printf 缓冲区, 让主机看到打印结果.
    cudaDeviceSynchronize();

    // 本示例未检查 CUDA API 返回值; 调试失败时应检查启动错误及同步返回值.
    return EXIT_SUCCESS;
}
