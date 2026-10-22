配置:
cmake / Ninja (sudo apt install build-essential cmake ninja-build)
ms-vscode.cpptools, vscode C/C++ 扩展的 IntelliSense, 用于代码跳转(结合compile_commands.json)
nvidia.nsight-vscode-edition, Nsight Visual Studio Code Edition, 用于单步kernel
    vscode加这个extension只能在WSL/Linux单步kernel, Windows上单步kernel要用VS

Windws/Linux编译:
在root下面cmake配置费时, 每个app独立编译, 见run.sh

todo:


next:


list见doc/learning/xx.md

已整理好 [初学者学习清单](C:/code/cuda-samples/docs/learning/README.md), 包含推荐顺序, 学习重点, 以及你旧清单中的 **14 个已读样例**对照.

按你之前的进度, 建议接着看:

1. `reduction`: 并行归约, shared memory, warp shuffle.
2. `transpose`: 合并访存, bank conflict.
3. `simpleAtomicIntrinsics`: 原子操作.
5. `simpleCooperativeGroups`: 线程分组与同步.

迁移也已完成:

- **7 个当前源码文件**补入了适配新版的学习注释, 未改变执行逻辑.
- 原始阅读记录, **13 份带注释源码**及最新提交完整补丁已[归档](C:/code/cuda-samples/docs/learning/legacy-8efda95/README.md).
- 已核对归档内容与旧提交一致, 清单链接有效.

旧仓库未删除; 此次保存的是最新提交相关笔记, 不是整个旧仓库历史.

**已经直接迁入当前仓库, 修改还未提交. ** 迁移来源是旧仓库最新提交 `8efda95`.

| 样例 | 迁移结果 / 冲突处理 |
|---|---|
| `matrixMul` | 已迁入 shared memory, 计时, 矩阵行列尺寸注释 |

**当前没有同名样例, 只能归档的: **

- `cppIntegration`
- `cppOverload`
- `matrixMulCUBLAS`
- `simpleOccupancy`
- `concurrentKernels`
- `bandwidthTest`: 最新提交没有源码注释, 只保留阅读记录.

**没有未解决的合并冲突. ** 采用逐项适配, 没有直接套用旧补丁; 当前 7 份源码只增加注释, 未改变执行逻辑. 已检查差异格式及归档一致性, 未重新编译.

详细记录见 [迁移对照与学习清单](C:/code/cuda-samples/docs/learning/README.md). [历史归档](C:/code/cuda-samples/docs/learning/legacy-8efda95/README.md)保留了原始注释全文, 13 份源码及完整提交补丁. 旧仓库尚未删除.


找到了, 已补进[学习清单](C:/code/cuda-samples/docs/learning/README.md), 包含具体源码入口和差异.

| 旧样例 | 当前接近的 example | 替代程度 |
|---|---|---|
| `cppIntegration` | `scan` | 可以学习 `.cpp/.cu` 混编, 接口封装和 kernel 调用 |
| `cppOverload` | `MC_EstimatePiP` + `interval` | 分别包含 kernel 属性查询, 缓存配置; 不能完整替代重载演示 |
| `matrixMulCUBLAS` | `UnifiedMemoryStreams` | 有 cuBLAS 调用, 但做的是 **GEMV(矩阵乘向量)**; 未找到 GEMM 对应项 |
| `simpleOccupancy` | `simpleAWBarrier` | **两个核心 occupancy API 都有**; 先看 launch 配置部分 |
| `concurrentKernels` | `simpleStreams` + `graphMemoryNodes` | 分别学习多 stream 工作, event 跨 stream 依赖; 不是原来的并发计时测试 |
| `bandwidthTest` | `simpleMultiCopy` | 有 H2D/D2H 计时和吞吐比较; 没有完整覆盖 D2D 等测试模式 |

建议先看 **`scan` 的接口组织 -> `simpleAWBarrier` 的配置代码 -> `simpleMultiCopy` 的计时部分**.

这些替代项的算法和参数不同, 旧注释没有直接塞进它们的源码, 已通过清单关联原笔记. 本次检查了源码和链接, 未编译运行.