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
#include <cassert>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

const char *sampleName = "simpleAssert";

// Auto-Verification Code
bool testResult = true;

// Each thread computes its global index and asserts it is below N. Threads with
// gtid >= N trip the assertion, which aborts the kernel and surfaces on the host
// as cudaErrorAssert at the next synchronization point.
// 本例故意触发设备端断言, 演示 CPU 如何检测 GPU 执行错误.
// assert 需要保持启用: 若编译时定义了 NDEBUG, 断言检查会被移除.
__global__ void simpleAssertKernel(int N)
{
    // 一维全局线程编号 = 线程块编号 * 每块线程数 + 块内线程编号.
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // 条件为假时报告设备端断言失败; 它不是让越界线程安全跳过的 if 判断.
    assert(gtid < N);
}

int main(int argc, char **argv)
{
    // 启动 2 个 block, 每个 block 32 个线程, 共 64 个线程, 编号为 0~63.
    int         Nblocks  = 2;
    int         Nthreads = 32;
    cudaError_t error;

    printf("%s starting...\n\n", sampleName);

    // Select device 0 as the active GPU
    int devID = 0;
    // 选择当前主机线程后续 CUDA 操作使用的 GPU, 设备编号从 0 开始.
    cudaSetDevice(devID);

    // Query compute capability (major.minor) and number of SMs on the device
    // major.minor 是计算能力版本; smCount 是 GPU 的流式多处理器数量.
    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devID);

    // Print device info
    printf("GPU Device %d: with compute capability %d.%d and Number of SMs %d\n\n", devID, major, minor, smCount);

    // Kernel configuration, where a one-dimensional
    // grid and one-dimensional blocks are configured.
    // 本例故意触发设备端 assert; 观察下面同步调用返回的错误.
    // dim3 未指定的 y, z 维度默认为 1, 此处grid和block为一维.
    dim3 dimGrid(Nblocks);
    dim3 dimBlock(Nthreads);

    printf("Launch kernel to generate assertion failures\n");
    // 传入 N=60, 编号 60~63 的四个线程不满足断言条件, 故意制造失败.
    // 核函数启动相对 CPU 是异步的, 需要在后面的同步点检查执行结果.
    simpleAssertKernel<<<dimGrid, dimBlock>>>(60);

    // Synchronize (flushes assert output).
    printf("\n-- Begin assert output\n\n");
    // 等待设备执行并取得错误状态, 同时刷新设备端断言输出.
    // 多线程并行执行, 断言消息的打印顺序不保证按线程编号排列.
    error = cudaDeviceSynchronize();
    printf("\n-- End assert output\n\n");

    // Check for errors and failed asserts in asynchronous kernel launch.
    if (error == cudaErrorAssert) {
        printf("Device assert failed as expected, "
               "CUDA error message is: %s\n\n",
               cudaGetErrorString(error));
    }

    // 本例的成功标准是捕获预期的 cudaErrorAssert, 而不是核函数正常完成.
    // 因此看到断言失败后再输出 OK, 表示示例验证通过.
    testResult = error == cudaErrorAssert;

    printf("%s completed, returned %s\n", sampleName, testResult ? "OK" : "ERROR!");
    // 设备端断言失败后不能直接继续正常使用该 CUDA 上下文; 本例验证后退出.
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
