#!/usr/bin/env bash
set -euo pipefail

if [[ "$OSTYPE" == linux* ]]; then
    # Linux:
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89

    cmake --build build -j 8

    [ -f build/compile_commands.json ] && cp build/compile_commands.json ../../../compile_commands.json

    echo "running..."
    ./build/deviceQuery
fi

if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]]; then
    # Windows: Run from Git Bash with the x64 MSVC environment loaded.
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER=cl \
        -DCMAKE_CXX_COMPILER=cl \
        -DCMAKE_CUDA_HOST_COMPILER=cl \
        -DCMAKE_CUDA_ARCHITECTURES=89

    cmake --build build -j 8

    [ -f build/compile_commands.json ] && cp build/compile_commands.json ../../../compile_commands.json

    echo "running..."
    ./build/deviceQuery.exe
fi

: <<'COMMENT'

log example:

C:\code\cuda-samples\cpp\1_Utilities\deviceQuery\build\deviceQuery.exe Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 4070 Laptop GPU"
  CUDA Driver Version / Runtime Version          13.4 / 13.4
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 8188 MBytes (8585281536 bytes)
  (036) Multiprocessors, (128) CUDA Cores/MP:    4608 CUDA Cores
  GPU Max Clock rate:                            1230 MHz (1.23 GHz)
  Memory Clock rate:                             8001 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 33554432 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Total number of registers available per sm:    65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of blocks per multiprocessor:   24
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 13.4, CUDA Runtime Version = 13.4, NumDevs = 1
Result = PASS


./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Xavier"
  CUDA Driver Version / Runtime Version          11.4 / 11.4
  CUDA Capability Major/Minor version number:    7.2
  Total amount of global memory:                 30991 MBytes (32496091136 bytes)
  (008) Multiprocessors, (064) CUDA Cores/MP:    512 CUDA Cores
  GPU Max Clock rate:                            1377 MHz (1.38 GHz)
  Memory Clock rate:                             675 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 524288 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        98304 bytes
  Total number of registers available per block: 65536
  Total number of registers available per sm:    65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of blocks per multiprocessor:   32
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            Yes
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.4, CUDA Runtime Version = 11.4, NumDevs = 1
Result = PASS

COMMENT