#!/usr/bin/env bash
set -euo pipefail

if [[ "$OSTYPE" == linux* ]]; then
    # Linux:
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCMAKE_CUDA_FLAGS_DEBUG="-g -G"

    cmake --build build -j 8

    [ -f build/compile_commands.json ] && cp build/compile_commands.json ../../../compile_commands.json

    echo "running..."
    ./build/simpleZeroCopy
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
    ./build/simpleZeroCopy.exe
fi
