# GPU 共享三：井水不犯河水

在最后，我们来讨论多个任务共享 GPU 时，算力和显存是否可以做到井水不犯河水。当前开源的方案主要是腾讯 TKEStack 的 GaiaGPU。VirtAI 也有一个类似的功能的社区版，我们也会稍作讨论。

## API 劫持

我们先来看看 CUDA 任务使用显存的基本逻辑吧。我们以申请一块 NxN 的矩阵为例。

0. 可以预先计算一下 NxN 的 float32 矩阵需要多少空间：`uint32 bytes = N * N * sizeof(float);`
1. 查询整块 GPU 的显存总量或当前剩余的 GPU 显存总量
2. （如果剩下的显存量够多）利用 CUDA Runtime API 或者 CUDA Driver API 向 GPU 申请 bytes 那么多的显存
3. 确认申请显存的操作成功

那么当多个任务同时在一块 GPU 上进行显存申请的时候，我们希望做到和上述步骤不同的地方是：**2. 查询显存总量/剩余显存总量**时，返回的不希望是整块 GPU 的显存相关信息，而是限制之后的相关信息。举个例子，一块 GPU 一共有 12Gi 显存，我们分配给任务 A 5Gi 显存，给任务 B 3Gi 显存。当任务 A 中的进程调用 `cuDeviceTotalMem` 时，应该返回 5Gi。假设任务 A 中的进程已经使用了 1.5 Gi 的显存，那么当调用 `cuMemGetInfo` 时，应该返回 3.5 Gi。

而如果有的用户程序不经过步骤 2，直接执行步骤 3 呢？显然，我们需要根据上述逻辑对 “**如果剩下的显存量够多**” 作出判断。

那么要修改 CUDA 函数的实现，一般可以走 `LD_PRELOAD` 或 `LD_LIBRARY_PATH` 两种方式。前者的模式适用范围有限，无法进一步劫持由 Runtime API 对 Driver API 的调用。因此 TKEStack 采取修改 `libcuda` 并通过 `LD_LIBRARY_PATH` 加载。具体的代码可以参看：[vcuda-controller](https://github.com/tkestack/vcuda-controller)。

## 负反馈调节？

在解决了显存问题之后，我们来看看算力是否也能通过劫持做到限制？

在腾讯的 [GaiaGPU](https://ieeexplore.ieee.org/abstract/document/8672318) 这篇文章中，号称可以做到限制。其中心思想，在我看来是一种朴素的控制理论：负反馈调节。

每次发起 CUDA kernel 的时候，都需要检查一下当前任务对 GPU 的使用率，并对本次 CUDA kernel 会增加的 GPU 使用率作出估计。如果预计本次 CUDA kernel 会使得 GPU 使用率超标，则延缓 kernel 的运行，直到当前的 GPU 使用率下降至允许本次 CUDA kernel 的运行之后。然后这样做，无法避免多个任务的 context 切换带来的 overhead。

劫持的工程实现是做在 vcuda-controller 上的。

## 逆向工程

这里简单讲一下 VirtAI 的社区版跟 TKEStack 的异同之处。

VirtAI 社区版其实只是供社区使用的一个安装包，里面不含任何代码。我是通过抓去安装包内的信息大致作出的推测。

1. 两者都采取了劫持 CUDA API 的方式，但 VirtAI 不仅劫持了 Driver API，还同时劫持了 Runtime API
2. VirtAI 的将所有的 API 用 rpc 进行了包装，使之可以运行在调用 API 的进程之外，这样也就解释了为什么 GPU 节点上会有 orind 这个 daemon 进程存在
3. VirtAI 号称实现了类似 MPS 的多 CUDA 进程无 context 切换，这是怎么操作的尚还不知晓

## 结语

可见利用 API 劫持确实是可以玩出许多花样来。但是我希望大家还是可以回顾一下第一篇，仔细考虑自己的集群和用户是否真的需要 GPU 共享再做决定。还是那句话，别人有的，不一定是最适合你的。