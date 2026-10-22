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

/*
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are inserted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 */

// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions

// 每个 GPU 线程为一个元素加上 inc_value; 本例每个元素最终应为 26.
__global__ void increment_kernel(int *g_data, int inc_value)
{
    // 本例 n 恰好被每块线程数整除, 因此线程数等于元素数; 修改规模时需考虑边界检查.
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

// 数据拷回主机后, CPU 逐元素校验, 遇到第一个不匹配值就返回失败.
bool correct_output(int *data, const int n, const int x)
{
    for (int i = 0; i < n; i++)
        if (data[i] != x) {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }

    return true;
}

int main(int argc, char *argv[])
{
    int            devID;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    int n      = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value  = 26;

    // allocate host memory
    int *a = 0;
    // 分配页锁定主机内存, 为主机与设备之间的异步 DMA 传输提供条件.
    // checkCudaErrors 是示例辅助函数, 用于检查 CUDA 调用是否成功.
    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a = 0;
    checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    // cudaMemset 按字节填充: 255 生成全 0xFF 的位模式, 并非将每个 int 设为 255.
    // 后续主机到设备的拷贝会覆盖此初值.
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    // 一维block, 512 thread
    dim3 threads = dim3(512, 1);
    // 一维grid, 16*1024*1024/512=32768个block.
    dim3 blocks  = dim3(n / threads.x, 1);

    // create cuda event handles
    // 两个事件作为同一 stream 中的计时边界, 记录执行到对应位置时的时间.
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // 计时前等待之前的初始化操作结束, 避免把它们混入本次测量.
    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // CPU timer 测量提交调用所花的时间,
    // CUDA events 测量 GPU stream 上 start 到 stop 的时间, 两者含义不同.
    // 本例 a 来自 cudaMallocHost(pinned memory), 使用异步拷贝让 CPU 继续工作.
    // 不能泛化为所有带 Async 的 API 都绝不阻塞; 行为还取决于内存类型和方向.
    // asynchronously issue work to the GPU (all to stream 0)
    checkCudaErrors(cudaProfilerStart());
    sdkStartTimer(&timer);
    // 以下操作全部进入同一个默认 stream, 按顺序执行, 并不会彼此重叠.
    // 此处演示的是 CPU 与 GPU 工作重叠, 而不是多条 stream 间的并行.
    cudaEventRecord(start, 0);
    // H2D: 把主机数组中的零传到设备; 提交后 CPU 可以继续执行后续代码.
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    // 启动参数依次为: 块数, 每块线程数, 动态共享内存字节数, stream.
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    // D2H: 在前面的核函数完成后拷回结果; 确认完成前 CPU 不应读取或改写 a.
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    // stop 排在结果拷贝之后, 因此该事件完成也表示主机结果已经可用.
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    checkCudaErrors(cudaProfilerStop());

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    // 非阻塞查询事件: 未完成时返回 cudaErrorNotReady, CPU 可利用这段时间工作.
    // 此处用计数演示忙轮询; 循环次数取决于运行时序, 不是固定值, 也不是 GPU 线程数.
    // 当前循环只区分 NotReady; 其他错误也会使循环退出, 并不等同于成功完成.
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    // GPU 事件耗时单位为毫秒, 覆盖 H2D, 核函数和 D2H 所在的整个时间区间.
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // check the output for correctness
    bool bFinalResults = correct_output(a, n, value);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    // 页锁定主机内存用 cudaFreeHost 释放, 设备内存用 cudaFree 释放.
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
