# 重峦叠重般的 GPU 利用率

本文会讨论在一个集群中的 GPU 使用率，但我们会一步一步来，尽量不直接引入一些特别专业的概念。

*GPU 的使用范围还包括编解码等，这里我们只讨论 GPU 的流处理器（SM）用于通用计算时的利用率*

## First Impression

如果我们对 GPU 一无所知，只是知道这是个***不可被任务共享[1]***的设备，那我们如何统计一个集群上的 GPU 使用率呢？

一般来讲，统计使用率，我们需要先设定一个时间跨度。那么现在假设从 `T0` 时刻到 `T1` 时刻。

我们再做一个假设：GPU 只有两种状态，即正在被使用和闲置。

这样似乎是足够我们得到一下公式：

`Util_GPU_Cluster = Integrate[u(t), {T0, T1}]/(N*(T1-T0))`

这里，`N` 为该集群上 GPU 的总数，`u(t)` 是 `t` 时刻该集群上正在被使用的 GPU 的个数。（有知道怎么在 markdown + github 上写积分表达式的麻烦请告诉我。）

如果我们有针对每一个任务的监控，那么可以通过对于集群上任务的状态来判定 GPU 的状态，从而描绘出 `u(t)`；如果没有，那么至少会有任务是否调度成功的判定，我们只需要再添加一个”*任务调度成功即开始运行*“的假设即可。

## Zoom In

刚才的演绎，明显是给予 GPU 只有 ON/OFF 两个状态的假定。想必很多了解 Nvidia GPU 的人都会摇头：naive。

即使不会写 CUDA 程序，很多人也知道通过 `nvidia-smi` 可以获取当前 GPU 的使用率：GPU Util.

没错，Nvidia 在描述一块 GPU 的使用率的时候，采用了百分比的方式。也就是说，任意时刻，一块 GPU 的使用率并不是 0/1 这样的取值，而是从 0% 到 100% 的离散划分。这样一来，回顾上面的公式，`u(t)` 不再返回 0 或 1，而是返回 0.00 到 1.00 之间的值。 

## Enough?

似乎大功告成了？

且慢，请看 NVML 关于 GPU 利用率的解释：

[nvmlUtilization_t](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html#structnvmlUtilization__t)

> `unsigned int nvmlUtilization_t::gpu [inherited]`

**Percent of time over the past sample period during which one or more kernels was executing on the GPU.**

啊，原来 Nvidia 统计一块 GPU 使用率的时候，采取了跟我们一开始类似的统计方式。NVML 并没有统计 GPU 的有多少流处理器（SM）被使用，而是采取时间切片的方式，计算了在过去的 K 个时间切片中，有百分之几的时间切片里，GPU 有至少一个 kernel 函数正在被运行。

这样一来，我们可以想一想，之前根据 GPU Util. 统计出来的集群 GPU 使用率是高估了还是低估了？

嗯，是高估了。因为即便是 100% 的 GPU 使用率，也有可能只有 1 个流处理器（SM）被使用。

如果我们想要做到更精确的统计，那该怎么办呢？

## Alternatives

第一个方法比较 hacky，实话说也尚无实验做确凿的验证。但也不妨一试。

我们可以想象，想要驱动更多的流处理器，就需要更多的电力。而平时玩游戏的时候我们也能发现，渲染桌面的时候，GPU 耗电低；渲染游戏画面的时候，GPU 耗电高。这样一来，我们不妨通过 GPU 功率来衡量流处理器的使用率，当然是否线性，是否过原点（应该不过）我们尚且需要验证。

按照这种想法 `u(t) = p(t)`，`p(t)` 即为 `t` 时刻 GPU 的功率占最大功率的百分比。需要注意的是，如果 `p(t)` 是针对一个（混部）集群上的所有 GPU 进行描述的话，需要先做百分比，再做平均。

相比很多人并不喜欢上面这个方案。其实 NVML 中还有另一个函数可以一试：

[nvmlDeviceGetVgpuProcessUtilization](https://docs.nvidia.com/deploy/nvml-api/group__nvmlUtil.html#group__nvmlUtil_1gded837e47351b26f958aa083f8d004ff)

通过 `nvmlDeviceGetVgpuProcessUtilization`，我们可以获取该 GPU 上所有进程对流处理器的使用率：

[**unsigned int smUtil: SM (3D/Compute) Util Value.**](https://docs.nvidia.com/deploy/nvml-api/structnvmlVgpuProcessUtilizationSample__t.html#structnvmlVgpuProcessUtilizationSample__t)

在进行简单求和之后，我们就可以得到整块 GPU 的 SM 使用率。

当然在不支持 vGPU （GRID）的 GPU 上是否可以使用该函数，还需要进一步验证。

## Finally

到这里，我们已经至少讲了 3 个层次的集群 GPU 使用率了：

1. 二值集群 GPU 使用率：`U_b`
2. 时间切片集群 GPU 使用率：`U_t`
3. 集群 GPU SM/功率 使用率：`U_p`

从衡量 GPU 算力使用的角度看，三种使用率在精确度上的排序应该是：

`U_b < U_t < U_p`

第三种方案应是最精确的。

那么是否我们就应该统一使用第三种方案来对集群上的 GPU 使用率进行统计呢？

其实每一种统计方式都能反映出不同的问题：

二值使用率尽管忽略了 GPU 的使用并非 ON/OFF，却很好地衡量了集群对于 GPU 任务的调度状况。例如，我们想要比较两个调度器对 GPU 任务的调度表现，没有必要把 GPU 任务对 GPU 的使用率也考虑在内。

时间切片使用率是 Nvidia 官方认可的统计方式，应该是最广受认可的一种。一般对集群 GPU 使用率的统计都会采用这种标准。

那什么时候用更精确的衡量标准呢？嗯，离职前给对接人埋坑的时候！

当然是开玩笑的。一般而言，GPU 任务对 GPU 的使用是独占的。然而对于一些具有 GPU 共享的集群，一块 GPU 上有一个任务或是多个任务往往无法通过时间切片使用率的统计方式体现。因此，流处理器使用率的方式就可以派上用场了。

## TBD

接下来，我还会把一些如何提高 GPU 集群使用率的思考也写下来，请大家批评指正。

[1] 这是个假设，在原生 Kubernetes 集群上单个 GPU 不可被多个容器共享。（共享方案今后会讨论）

