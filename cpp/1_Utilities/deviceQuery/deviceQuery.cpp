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

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

// std::system includes

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <memory>
#include <string>

int   *pArgc = NULL;
char **pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T> inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

    if (CUDA_SUCCESS != error) {
        fprintf(
            stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n", error, __FILE__, __LINE__);

        exit(EXIT_FAILURE);
    }
}

#endif /* CUDART_VERSION < 5000 */


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query App version: cuda-samples v13.4\n");
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int         deviceCount = 0;
    cudaError_t error_id    = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }
    else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        // Driver Version 表示驱动支持的 CUDA 版本, Runtime Version 表示当前运行时版本.
        // 版本整数按 major * 1000 + minor * 10 编码; 计算能力则描述 GPU 架构能力.
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000,
               (driverVersion % 100) / 10,
               runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        // 学习笔记(迁自 LearnCudaSamples): global memory 是 GPU 可用的全局内存.
        // 独显通常对应板载显存; 旧 Xavier 平台则与 CPU 共享物理 DRAM.
        char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(msg,
                  sizeof(msg),
                  "  Total amount of global memory:                 %.0f MBytes "
                  "(%llu bytes)\n",
                  static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                  (unsigned long long)deviceProp.totalGlobalMem);
#else
        snprintf(msg,
                 sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes "
                 "(%llu bytes)\n",
                 static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                 (unsigned long long)deviceProp.totalGlobalMem);
#endif
        printf("%s", msg);

        // Multiprocessors 指 SM(Streaming Multiprocessors), 不是 CUDA core 数.
        // 每个 SM 的 core 数由 helper 中的架构映射表推算, 并非运行时直接查询值.
        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        int clockRate;
        checkCudaErrors(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev));
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f "
               "GHz)\n",
               clockRate * 1e-3f,
               clockRate * 1e-6f);
#if CUDART_VERSION >= 5000
        int memoryClockRate;
#if CUDART_VERSION >= 13000
        checkCudaErrors(cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, dev));
#else
        memoryClockRate = deviceProp.memoryClockRate;
#endif
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }

#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the
        // CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }

#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
               "%d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D,
               deviceProp.maxTexture2D[0],
               deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0],
               deviceProp.maxTexture3D[1],
               deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0],
               deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
               "layers\n",
               deviceProp.maxTexture2DLayered[0],
               deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);

        // shared memory 和寄存器既有单个 block 的上限, 也有整个 SM 的资源预算.
        // 同一 SM 上驻留的多个 block 共享这些资源, 因此资源用量会限制 occupancy.
        printf("  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        // 寄存器数量以 32-bit 寄存器为单位, 不是字节数.
        printf("  Total number of registers available per sm:    %d\n", deviceProp.regsPerMultiprocessor);
        // Warp 是线程执行的基本分组; 下面的 SM 上限描述同时驻留的线程和 block 数.
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of blocks per multiprocessor:   %d\n", deviceProp.maxBlocksPerMultiProcessor);
        // block 的 x*y*z 还必须不超过 maxThreadsPerBlock, 不能只检查各维度上限.
        // grid 维度限制一次 launch 的规模, 不代表所有 block 能同时驻留在 GPU 上.
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %zu bytes\n", deviceProp.textureAlignment);
        int gpuOverlap;
        checkCudaErrors(cudaDeviceGetAttribute(&gpuOverlap, cudaDevAttrGpuOverlap, dev));
        printf("  Concurrent copy and kernel execution:          %s with %d copy "
               "engine(s)\n",
               (gpuOverlap ? "Yes" : "No"),
               deviceProp.asyncEngineCount);
        int kernelExecTimeout;
        checkCudaErrors(cudaDeviceGetAttribute(&kernelExecTimeout, cudaDevAttrKernelExecTimeout, dev));
        printf("  Run time limit on kernels:                     %s\n", kernelExecTimeout ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
               deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        // UVA 统一主机和设备的虚拟地址空间; Managed Memory 另外提供托管内存能力.
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Managed Memory:                %s\n", deviceProp.managedMemory ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               deviceProp.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");
        // The property cooperativeMultiDeviceLaunch is deprecated in CUDA 13.0
#if CUDART_VERSION < 13000
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
#endif
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               deviceProp.pciDomainID,
               deviceProp.pciBusID,
               deviceProp.pciDeviceID);

        const char *sComputeMode[] = {"Default (multiple host threads can use ::cudaSetDevice() with device "
                                      "simultaneously)",
                                      "Exclusive (only one host thread in one process is able to use "
                                      "::cudaSetDevice() with this device)",
                                      "Prohibited (no host thread can use ::cudaSetDevice() with this "
                                      "device)",
                                      "Exclusive Process (many threads in one process is able to use "
                                      "::cudaSetDevice() with this device)",
                                      "Unknown",
                                      NULL};
        int         computeMode;
        checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, dev));
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    // 多 GPU 时查询有方向的 P2P 访问能力; 这里只查询, 不启用 peer access.
    if (deviceCount >= 2) {
        cudaDeviceProp prop[64];
        // 保存通过下方初筛的 GPU 编号, 再逐对查询实际 P2P 能力.
        int            gpuid[64]; // we want to find the first two GPUs that can support P2P
        int            gpu_p2p_count = 0;

        for (int i = 0; i < deviceCount; i++) {
            checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                // on Windows (64-bit), the Tesla Compute Cluster driver for windows
                // must be enabled to support this
                && prop[i].tccDriver
#endif
            ) {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int can_access_peer;

        if (gpu_p2p_count >= 2) {
            for (int i = 0; i < gpu_p2p_count; i++) {
                for (int j = 0; j < gpu_p2p_count; j++) {
                    if (gpuid[i] == gpuid[j]) {
                        continue;
                    }
                    checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                           prop[gpuid[i]].name,
                           gpuid[i],
                           prop[gpuid[j]].name,
                           gpuid[j],
                           can_access_peer ? "Yes" : "No");
                }
            }
        }
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char        cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
    snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#endif
    sProfileString += cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
    snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#endif
    sProfileString += cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
    sProfileString += cTemp;
    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

    printf("Result = PASS\n");

    // finish
    exit(EXIT_SUCCESS);
}
