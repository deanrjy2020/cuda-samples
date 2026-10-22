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

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

/* Add two vectors on the GPU */
// GPU 直接访问 mapped host memory 中的 a, b, c, 不需要单独的 device buffer 或显式 cudaMemcpy.
__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    // 每个 thread 负责一个元素; 相邻 thread 访问相邻元素, 有利于 coalesced access.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // grid 向上取整时可能产生多余 thread, 边界检查防止越界访问.
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Allocate generic memory with malloc() and pin it laster instead of using
// cudaHostAlloc()
// 默认使用 cudaHostAlloc; --use_generic_memory 切换到 malloc + cudaHostRegister.
bool bPinGenericMemory = false;

// Macro to aligned up to the memory size in question
// 此处保留原有 alignment macro: size 为 2 的幂时, 将地址向上取整到 size 的倍数.
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

int main(int argc, char **argv)
{
    int            n, nelem, deviceCount;
    int            idev   = 0; // use default device 0
    char          *device = NULL;
    unsigned int   flags;
    size_t         bytes;
    // a/b/c 供 CPU 访问; d_a/d_b/d_c 是同一批 host allocation 对应的 device pointer.
    // a_UA/b_UA/c_UA 保存 malloc 返回值, 后续 free 必须使用这些原始地址.
    float         *a, *b, *c;          // Pinned memory allocated on the CPU
    float         *a_UA, *b_UA, *c_UA; // Non-4K Aligned Pinned memory on the CPU
    float         *d_a, *d_b, *d_c;    // Device pointers for mapped memory
    float          errorNorm, refNorm, ref, diff;
    cudaDeviceProp deviceProp;

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printf("Usage:  simpleZeroCopy [OPTION]\n\n");
        printf("Options:\n");
        printf("  --device=[device #]  Specify the device to be used\n");
        printf("  --use_generic_memory (optional) use generic page-aligned for system "
               "memory\n");
        return EXIT_SUCCESS;
    }

    /* Get the device selected by the user or default to 0, and then set it. */
    if (getCmdLineArgumentString(argc, (const char **)argv, "device", &device)) {
        cudaGetDeviceCount(&deviceCount);
        idev = atoi(device);

        if (idev >= deviceCount || idev < 0) {
            fprintf(stderr, "Device number %d is invalid, will use default CUDA device 0.\n", idev);
            idev = 0;
        }
    }

    // if GPU found supports SM 1.2, then continue, otherwise we exit
    if (!checkCudaCapabilities(1, 2)) {
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "use_generic_memory")) {
#if defined(__APPLE__) || defined(MACOSX)
        bPinGenericMemory = false; // Generic Pinning of System Paged memory is not
                                   // currently supported on Mac OSX
#else
        bPinGenericMemory = true;
#endif
    }

    if (bPinGenericMemory) {
        printf("> Using Generic System Paged Memory (malloc)\n");
    }
    else {
        printf("> Using CUDA Host Allocated (cudaHostAlloc)\n");
    }

    checkCudaErrors(cudaSetDevice(idev));

    /* Verify the selected device supports mapped memory and set the device
       flags for mapping host memory. */

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, idev));

#if CUDART_VERSION >= 2020

    // 先检查所选 GPU 是否支持 mapped host memory, 不支持则跳过本 sample.
    if (!deviceProp.canMapHostMemory) {
        fprintf(stderr, "Device %d does not support mapping CPU host memory!\n", idev);

        exit(EXIT_SUCCESS);
    }

    // 指定 host memory mapping 相关的 device flag; 实际 allocation 和 mapping 在后面完成.
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
#else
    fprintf(stderr,
            "CUDART version %d.%d does not support "
            "<cudaDeviceProp.canMapHostMemory> field\n",
            ,
            CUDART_VERSION / 1000,
            (CUDART_VERSION % 100) / 10);

    exit(EXIT_SUCCESS);
#endif

#if CUDART_VERSION < 4000

    if (bPinGenericMemory) {
        fprintf(stderr,
                "CUDART version %d.%d does not support <cudaHostRegister> function\n",
                CUDART_VERSION / 1000,
                (CUDART_VERSION % 100) / 10);

        exit(EXIT_SUCCESS);
    }

#endif

    /* Allocate mapped CPU memory. */
    // 两种方式都在 host 端分配内存并映射给 GPU.
    // 1. malloc + cudaHostRegisterMapped: 注册已有内存, 释放时先 unregister 再 free.
    // 2. cudaHostAllocMapped: CUDA 分配 pinned + mapped 内存, 用 cudaFreeHost 释放.
    // 本例手动对齐的实现不代表所有 CUDA 版本的注册 API 都要求用户做 4K 对齐.
    // 旧测试平台 Xavier 的 CPU/GPU 共享物理 DRAM; 当前独显访问 host memory
    // 通常需要经过 PCIe. 省去显式 memcpy 不等于访问免费, 也不保证比 device memory 快.

    // 每个 array 有 1048576 个 float, 占 4 MiB; 输入 a/b 和输出 c 共 12 MiB 有效数据.
    nelem = 1048576;
    bytes = nelem * sizeof(float);

    if (bPinGenericMemory) {
#if CUDART_VERSION >= 4000
        // 注册已有的 host allocation, 并保留原指针供 free 使用.
        // 多分配 MEMORY_ALIGNMENT bytes, 保证向上调整地址后仍能容纳 bytes 的有效数据.
        a_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
        b_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
        c_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);

        // We need to ensure memory is aligned to 4K (so we will need to padd memory
        // accordingly)
        a = (float *)ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
        b = (float *)ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
        c = (float *)ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

        // 注册从对齐地址开始的有效范围, 将已有 host memory 变成 pinned + mapped memory.
        checkCudaErrors(cudaHostRegister(a, bytes, cudaHostRegisterMapped));
        checkCudaErrors(cudaHostRegister(b, bytes, cudaHostRegisterMapped));
        checkCudaErrors(cudaHostRegister(c, bytes, cudaHostRegisterMapped));
#endif
    }
    else {
#if CUDART_VERSION >= 2020
        // 直接分配 pinned + mapped host memory.
        flags = cudaHostAllocMapped;
        checkCudaErrors(cudaHostAlloc((void **)&a, bytes, flags));
        checkCudaErrors(cudaHostAlloc((void **)&b, bytes, flags));
        checkCudaErrors(cudaHostAlloc((void **)&c, bytes, flags));
#endif
    }

    /* Initialize the vectors. */

    // CPU 在 kernel launch 前写入输入; c 不需要初始化, kernel 会覆盖每个有效元素.
    for (n = 0; n < nelem; n++) {
        a[n] = rand() / (float)RAND_MAX;
        b[n] = rand() / (float)RAND_MAX;
    }

    /* Get the device pointers for the pinned CPU memory mapped into the GPU
       memory space. */

#if CUDART_VERSION >= 2020
    // 获取 GPU 可用的 pointer, 不执行数据拷贝, 也不额外分配 device memory.
    // host/device pointer 的数值可能相同, 但不应把数值相同当作本例的前提.
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));
#endif

    /* Call the GPU kernel using the CPU pointers residing in CPU mapped memory.
     */
    printf("> vectorAddGPU kernel will add vectors using mapped CPU memory...\n");
    // nelem=1048576: 4096 个 blocks, 每个 256 个 threads.
    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem / (float)block.x));
    vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem);
    // kernel launch 是异步的; CPU 必须等待 GPU 完成写入, 才能读取 mapped 输出 c.
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("vectorAddGPU() execution failed");

    /* Compare the results */

    printf("> Checking the results from vectorAddGPU() ...\n");
    // 累加误差和 reference 的平方和, 再取平方根, 得到相对 L2 error 的分子和分母.
    errorNorm = 0.f;
    refNorm   = 0.f;

    for (n = 0; n < nelem; n++) {
        ref  = a[n] + b[n];
        diff = c[n] - ref;
        errorNorm += diff * diff;
        refNorm += ref * ref;
    }

    errorNorm = (float)sqrt((double)errorNorm);
    refNorm   = (float)sqrt((double)refNorm);

    /* Memory clean up */

    printf("> Releasing CPU memory...\n");

    if (bPinGenericMemory) {
#if CUDART_VERSION >= 4000
        // unregister 解除注册, 不会释放 malloc allocation; 随后用原始 pointer 调用 free.
        checkCudaErrors(cudaHostUnregister(a));
        checkCudaErrors(cudaHostUnregister(b));
        checkCudaErrors(cudaHostUnregister(c));
        free(a_UA);
        free(b_UA);
        free(c_UA);
#endif
    }
    else {
#if CUDART_VERSION >= 2020
        // cudaHostAlloc 的 allocation 用 cudaFreeHost 释放; d_a/d_b/d_c 无需单独 cudaFree.
        checkCudaErrors(cudaFreeHost(a));
        checkCudaErrors(cudaFreeHost(b));
        checkCudaErrors(cudaFreeHost(c));
#endif
    }

    // 相对 L2 error 小于 1e-6 时通过; 这里验证结果, 没有测量 zero-copy 的性能.
    exit(errorNorm / refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE);
}
