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

/*
 * This sample illustrates the usage of CUDA streams for overlapping
 * kernel execution with device/host memcopies.  The kernel computes an
 * element-wise square of a float array, after which the result is copied
 * back to host (CPU) memory.  To increase performance, multiple
 * kernel/memcopy pairs are launched asynchronously, each pair in its
 * own stream.  A GPU can overlap a kernel and a memcopy as long as they
 * are issued in different streams.  Thus, if n pairs are launched, the
 * streamed approach can reduce the memcopy cost to the (1/n)th of a
 * single copy of the entire data set.
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5
 * microsecond precision.
 *
 */

// System includes
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// 当前配置: N=16777216, 一个array里面N个float, array占 64 MB, 共2个array
#define N (1 << 24) // ~16M elements
#define BLOCK 256   // threads per block
#define NUM_STREAMS 4   // number of concurrent streams

// Kernel: computes element-wise square of the input array
// grid和block都是一维的.
__global__ void square_kernel(const float* in, float* out, int n){
    // idx 是本次 kernel 输入范围内的 index, 即grid里面的global idx.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // grid 向上取整后可能包含多余 thread, 用边界检查避免越界访问.
    if (idx < n){
        // 每个 thread 计算一个元素的平方
        out[idx] = in[idx] * in[idx];
    }
}

// Run the full H2D memcopy + kernel + D2H memcopy in a single default stream.
// Returns elapsed time in milliseconds.
float run_default_stream(float* h_in, float* h_out, int n){
    float *d_in, *d_out;  // device input and output buffers
    size_t bytes = n * sizeof(float);

    // allocate device memory
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // create CUDA event handles for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 单 stream baseline: start -> H2D -> kernel -> D2H -> stop 按序执行.
    // event 区间不包含上面的 allocation, 也不包含后面的 resource cleanup.
    cudaEventRecord(start);

    // copy input to device, run kernel, copy result back
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    // grid (=65536) * BLOCK (=256) = N, 正好覆盖N个元素
    int grid = (n + BLOCK - 1) / BLOCK;
    square_kernel<<<grid, BLOCK>>>(d_in, d_out, n);
    // 此处使用同步 D2H copy, 返回后 CPU 才能读取 h_out 中的结果.
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // block until the event is actually recorded

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // release device resources
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// Run the same workload split across 4 streams so that H2D memcopies,
// kernel execution, and D2H memcopies can overlap on the device.
// Returns elapsed time in milliseconds.
// 把数据分为 4 份,
// 每份在自己的 stream 内依次执行 H2D -> kernel -> D2H; 跨 stream 有机会重叠.
// 本版使用 cudaStreamNonBlocking, 不适用旧笔记中依赖 legacy 默认 stream
// 隐式同步的计时解释, 也不能沿用旧版的 16M 元素, 512 threads 等数字.
// 补充上方旧笔记: 当前 N 仍是 16M, 但每个 block 是 256 threads.
// 不同 stream 只提供 overlap 机会; 实际效果取决于 copy engine 和 SM 等资源.
float run_multi_stream(float* h_in, float* h_out, int n){
    // 默认每份 16777216/4=4194304 个元素, 即 16 MiB; 当前实现要求 n 能被 NUM_STREAMS 整除.
    // 若修改 n 后存在 remainder, 下面的等分方式不会处理末尾剩余元素.
    int chunk = n/NUM_STREAMS;                      // number of elements per stream
    size_t chunk_bytes = chunk * sizeof(float);

    float *d_in[NUM_STREAMS], *d_out[NUM_STREAMS]; // per-stream device buffers
    cudaStream_t streams[NUM_STREAMS];

    // allocate and initialize an array of stream handles
    for(int i = 0; i < NUM_STREAMS; i++){
        // NonBlocking 表示不参与 legacy default stream 的隐式同步.
        // 它不保证所有 API 对 CPU 都不阻塞, 也不保证多个 kernel 同时执行.
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        // dev mem也开4组, 每组都有自己的stream
        cudaMalloc(&d_in[i], chunk_bytes);
        cudaMalloc(&d_out[i], chunk_bytes);
    }

    // create CUDA event handles for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start 只进入 streams[0]; 其他 stream 没有显式等待此 event.
    cudaEventRecord(start, streams[0]);

    // asynchronously launch 4 copies + kernels + copies, each in its own
    // stream so they can overlap on the device
    for(int i = 0; i < NUM_STREAMS; i++){
        // 各 stream 使用独立的 device buffer, 并访问互不重叠的 host chunk.
        int offset = i * chunk; // element offset into the host arrays for this stream

        // H2D: copy this stream's chunk to device (non-blocking on host)
        cudaMemcpyAsync(
            d_in[i],
            h_in + offset,
            chunk_bytes,
            cudaMemcpyHostToDevice,
            streams[i]
        );

        // kernel: will only start after the H2D copy in this stream completes
        // grid = 16384, 为原grid的1/4, 正好覆盖1/4个原array长度
        int grid = (chunk + BLOCK - 1) / BLOCK;
        // 第三个 launch 参数为 dynamic shared memory 大小, 第四个指定 stream.
        // kernel 只看到当前 chunk, 因此传入 chunk 而不是整个 array 的 n.
        square_kernel<<<grid, BLOCK, 0, streams[i]>>>(
            // d_in / d_out都已经是原array的1/4了.
            d_in[i], d_out[i], chunk
        );

        // D2H: copy result back; starts after the kernel in this stream finishes
        cudaMemcpyAsync(
            h_out + offset,
            d_out[i],
            chunk_bytes,
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    // 此处由 CPU 显式等待每个 stream, 之后才记录 stop; streams[0] 是数组第一项,
    // 并非 CUDA 默认 stream 0. 不要把它理解成默认 stream 自动等待其他 streams.
    // Wait for all streams to complete their work
    for(int i=0; i < NUM_STREAMS; i++) { 
       cudaStreamSynchronize(streams[i]);
     }
    // CPU 已等待所有 stream 完成, 再提交 stop; 区间可能包含 host 提交延迟和 GPU idle.
    // 这个测量不是单个 kernel 的耗时, 也没有为所有 stream 建立共同的 start barrier.
    cudaEventRecord(stop, streams[0]);

    // Sync with stream 0, which only completes once all other work
    // is done and the stop event is recorded
    cudaStreamSynchronize(streams[0]);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // release per-stream resources
    for(int i=0; i < NUM_STREAMS; i++){
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main(){

    printf("[ CUDA Sample: Streams ]\n\n");

    // Query compute capability and number of SMs on device 0
    int cuda_device = 0;
    int major = 0, minor = 0, smCount = 0;
    cudaDeviceGetAttribute(&major,   cudaDevAttrComputeCapabilityMajor, cuda_device);
    cudaDeviceGetAttribute(&minor,   cudaDevAttrComputeCapabilityMinor, cuda_device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount,    cuda_device);
    printf("GPU Device %d: with compute capability %d.%d and Number of SMs %d\n\n",
           cuda_device, major, minor, smCount);

    size_t bytes = N * sizeof(float);

    // allocate pinned host memory: the OS normally moves RAM pages around freely,
    // pinning locks these pages in place so the GPU can read them directly
    // without the data being moved away mid-transfer
    // cudaMallocHost 提供 pinned host memory, 使这里的 async H2D/D2H copy 可以与 CPU 工作重叠.
    float *h_in  = 0; // pointer to input data in host memory
    float *h_out = 0; // pointer to output data in host memory
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);

    // initialize input array
    for (int i = 0; i < N; i++)
        h_in[i] = (float)i;

    //////////////////////////////////////////////////////////////////////
    // time single-stream execution for reference
    float t1 = run_default_stream(h_in, h_out, N);

    //////////////////////////////////////////////////////////////////////
    // time execution with 4 streams
    // 复用相同输入, 将 multi-stream 结果写回 h_out; 当前 sample 没有逐元素验证结果.
    float t2 = run_multi_stream(h_in, h_out, N);

    printf("Single stream = %.3f ms\n", t1);
    printf("Multi-stream  = %.3f ms\n", t2);
    // 比值大于 1 才表示本次 multi-stream 更快; 单次测量和 Debug build 不代表稳定性能.
    printf("Speedup       = %.2fx\n",   t1 / t2);

    // release pinned host memory
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
