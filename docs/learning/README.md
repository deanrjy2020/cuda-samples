# CUDA Samples 学习清单

基于当前 CUDA 13.4 仓库, 以及旧仓库 LearnCudaSamples 最新提交 `8efda95ad18c5e06f02665b03d8973f4b7d7ed00` 的阅读记录整理.
"旧版已读"来自你原来的 Done with 清单, 不表示已在本机编译或复习过新版.

## 建议阅读顺序

已经读过的基础项可以快速复习. 按你原来的进度, **下一项先读 reduction**, 再补 transpose 和原子操作.
路径均指向当前仓库, 可以点击进入源码目录.

| 顺序 | 样例 | 你的进度 | 重点 / 小练习 |
| --- | --- | --- | --- |
| 5 | [matrixMul](../../cpp/0_Introduction/matrixMul) | 旧版已读, 注释已迁入 | shared memory tile, 数据复用, 同步和行列布局. 先看一个 tile, 再看完整循环; 本样例尺寸应符合其 tile 假设. |
| 6 | [reduction](../../cpp/2_Concepts_and_Techniques/reduction) | **旧清单的下一项** | 从最基础版本逐个看优化: block 内归约, shared memory, warp shuffle, 多 block 的结果合并; 先别一次看完所有变体. |
| 7 | [transpose](../../cpp/6_Performance/transpose) | 新增待学 | 比较直接转置和 shared-memory 分块; 理解合并访存与 bank conflict, 以及 padding 的作用. |
| 8 | [simpleAtomicIntrinsics](../../cpp/0_Introduction/simpleAtomicIntrinsics) | 新增待学 | 理解原子读改写, 区分"原子操作"和"线程同步"; 再思考为什么累加不总用 atomicAdd. |
| 12 | [simpleCooperativeGroups](../../cpp/0_Introduction/simpleCooperativeGroups) | 新增待学 | 在理解 reduction 后, 学习用 group 表达线程协作和同步范围. |
| 13 | [shfl_scan](../../cpp/2_Concepts_and_Techniques/shfl_scan) | 后续扩展 | 区分 scan 和 reduction, 学习 warp 内数据交换. |
| 14 | [simpleCudaGraphs](../../cpp/3_CUDA_Features/simpleCudaGraphs) | 后续扩展 | 在理解 streams/events 后再学 graph; 理解重复提交固定工作流的开销. |

第一轮可以暂缓 Tensor Core, 动态并行, 多 GPU, 图形互操作和 CUDA Tile 等专题.

## 旧清单与迁移结果

旧清单共有 **14 个已读样例**, 另外把 reduction 标成 todo. 以下包含当前没有同名目录的项, 避免把旧进度遗漏掉.

| 旧路径 | 当前对应 / 保存位置 | 处理说明 |
| --- | --- | --- |
| 0_Simple/cppIntegration | [旧源码](legacy-8efda95/sources/0_Simple/cppIntegration/cppIntegration.cu) | 当前没有同名样例; 保留线程处理 char/int2 的原注释. |
| 0_Simple/cppOverload | [旧源码](legacy-8efda95/sources/0_Simple/cppOverload/cppOverload.cu), [kernel](legacy-8efda95/sources/0_Simple/cppOverload/cppOverload_kernel.cuh) | 当前没有同名样例; 保留重载, kernel attributes, shared memory 用量笔记. |
| 0_Simple/matrixMul | cpp/0_Introduction/matrixMul | 迁入 tiling, 300 次计时, 行列尺寸笔记; 保留 LearnAI 外部学习目录提示. |
| 0_Simple/matrixMulCUBLAS | [旧源码](legacy-8efda95/sources/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp) | 当前没有同名样例; 完整保留 row-major / column-major 的数值例子. |
| 0_Simple/simpleOccupancy | [旧源码](legacy-8efda95/sources/0_Simple/simpleOccupancy/simpleOccupancy.cu) | 当前没有同名样例; 保留 occupancy API 笔记. 旧文的 1024 threads 推荐值是当时观察, 不能当作所有 GPU 的固定答案. |
| 1_Utilities/bandwidthTest | [旧阅读记录](legacy-8efda95/README.original.md) | 当前没有同名样例; 旧提交无源码注释改动. 旧速度仅作历史记录. |
| 6_Advanced/concurrentKernels | [旧源码](legacy-8efda95/sources/6_Advanced/concurrentKernels/concurrentKernels.cu) | 当前没有同名样例; 保留 events, 跨 stream 依赖, 归约和 Xavier 测量记录. |
| 6_Advanced/reduction(todo) | cpp/2_Concepts_and_Techniques/reduction | 接续学习起点, 未标为已读. |

## 缺失样例的相近替代

以下按当前源码中的实际 API 和构建方式对应, **不是重命名关系, 也不代表完整替代**. 本次仅检查源码, 没有编译运行这些替代项.

| 旧样例 / 学习目标 | 当前最接近的入口 | 覆盖范围与差异 |
| --- | --- | --- |
| cppIntegration: C++ 调用 CUDA 模块 | [scan/CMakeLists.txt](../../cpp/2_Concepts_and_Techniques/scan/CMakeLists.txt), [main.cpp](../../cpp/2_Concepts_and_Techniques/scan/main.cpp), [scan.cu](../../cpp/2_Concepts_and_Techniques/scan/scan.cu), [scan_common.h](../../cpp/2_Concepts_and_Techniques/scan/scan_common.h) | 同一 executable 混编 main.cpp, scan_gold.cpp 和 scan.cu; 从 main.cpp -> 公共声明 -> scan.cu 中的 host wrapper -> kernel launch 跟踪调用. 先学工程组织, 不必立刻读懂 scan 算法. 旧例的 char/int2 操作没有对应. |
| cppOverload: kernel attributes / cache 配置 | [MC_EstimatePiP/src/piestimator.cu](../../cpp/2_Concepts_and_Techniques/MC_EstimatePiP/src/piestimator.cu), [interval.cu](../../cpp/2_Concepts_and_Techniques/interval/interval.cu) | 前者调用 cudaFuncGetAttributes 并检查 maxThreadsPerBlock; 后者调用 cudaFuncSetCacheConfig. 只能覆盖旧笔记中的资源查询/缓存配置, 不能完整替代 kernel 重载演示. simpleAttributes 实际讲 stream 的 L2 access policy, 不是同一个主题. |
| matrixMulCUBLAS: 调用 cuBLAS 做矩阵乘法 | [UnifiedMemoryStreams.cu](../../cpp/0_Introduction/UnifiedMemoryStreams/UnifiedMemoryStreams.cu) | 可看 execute 中的 cublasSetStream, cublasDgemv, 以及 handle 的创建/销毁. 它做的是矩阵乘向量 GEMV, 并涉及 Unified Memory 和主机多线程; 当前 cpp 源码未找到 cuBLAS GEMM 调用, 原来的 GEMM 行列布局笔记仍应看归档. |
| simpleOccupancy: 自动选择 block size, 估算驻留 blocks | [simpleAWBarrier.cu](../../cpp/0_Introduction/simpleAWBarrier/simpleAWBarrier.cu) | 搜索 cudaOccupancyMaxPotentialBlockSize 和 cudaOccupancyMaxActiveBlocksPerMultiprocessor, 两者都在 launch 前使用. 最适合复习旧 API 笔记, 但 kernel 本身涉及 barrier/cooperative launch, 难于旧例; 先只读 launch 配置部分. |
| concurrentKernels: 并发 streams 和 event 依赖 | [simpleStreams.cu](../../cpp/0_Introduction/simpleStreams/simpleStreams.cu), [graphMemoryNodes.cu](../../cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu) | 先用 simpleStreams 理解多个 stream 的独立工作; 再只看 graphMemoryNodes 的 doNegateSquaresInStream, 学习 cudaEventRecord -> cudaStreamWaitEvent 及资源释放依赖. 后者是带异步分配的依赖链, 不是旧例"8 个独立 kernels 后汇总"的并发计时测试. |
| bandwidthTest: H2D / D2H / D2D 带宽 | [simpleMultiCopy.cu](../../cpp/0_Introduction/simpleMultiCopy/simpleMultiCopy.cu) | 搜索 memcpy_h2d_time, memcpy_d2h_time: 分别用 events 测 H2D/D2H, 输出吞吐并比较多 stream 重叠. 没有完整覆盖旧例的 D2D 和 pageable/pinned 等测试模式; transpose 的有效带宽也不能当作等价 D2D memcpy 指标. |

对于初学者: 先读 scan 的接口组织, simpleAWBarrier 的 launch 配置, simpleMultiCopy 的计时部分. 其余替代项复杂度更高, 可以只看对应 API 所在片段. 旧注释中的线程数量, 矩阵尺寸和平台实测值不适合直接复制到这些不同算法的源码中, 因此仍保留在原归档, 靠本表关联.

## 旧笔记的几个校正

归档保存原文, 以下校正帮助复习, 不修改历史记录:

- concurrentKernels: 归约循环 i=8 时是把槽位 8..15 加到 0..7; i=4 是 4..7 加到 0..3; i=2 是 2..3 加到 0..1; i=1 才把 1 加到 0. 旧注释中部分下标写错了一轮.
- concurrentKernels: 多个 stream 提供并发机会, 不保证 8 个 kernels 一定同时执行; clock ticks 也不能不考虑设备和运行条件而直接作为跨 GPU 的时间比较.
- matrixMulCUBLAS: 原笔记的核心恒等式是 Cᵀ = Bᵀ Aᵀ. 以交换 A/B 的方式调用 column-major GEMM 时, 还必须匹配 m/n/k 和 leading dimensions; 不能只机械交换两个指针.
