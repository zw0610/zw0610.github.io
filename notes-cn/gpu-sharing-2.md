# GPU 共享一：调度

GPU 共享的第二篇，我们会看看开源的集群调度器一般是怎么处理的 GPU 共享调度的。由于我对调度器的了解有限，在这里我们只对 Slurm 调度器以及 Kubernetes （阿里云的开源方案）进行简单的介绍。

## Slurm & GRES

Slurm 非常有“前瞻性”的提出了 "Generic Resources"（[GRES](https://slurm.schedmd.com/gres.html)）的概念，把一些非常见的资源归到了这伊磊。

默认的调度器无法处理 GRES，一般来说需要安装对应的 GRES Plugin。当然，如果在部署 Slurm 之前就已经在集群上装好了 NVML（一般随着驱动就会安装），Slurm 会利用 `select/cons_tres plugin` 自动进行检测并将（Nvidia）GPU 资源登记。 而在调度 GPU 任务时，Slurm 依旧使用 `CUDA_VISIBLE_DEVICES` 来对多 GPU 的节点进行以卡为单位的 GPU 资源隔离。每个任务在节点上被拉起时，Slurm 都会在 Epilog 和 Prolog 来配置 `CUDA_VISIBLE_DEVICES` 使得任务可以使用的 GPU 受到限制。这些配置都是在 Slurm 的 master 节点上完成的。

讲了这么多，依然说的是 GPU 的调度，而不是共享 GPU 的调度。接下来，我们来看看 Slurm 如何处理 GPU 共享调度。

除了 GPU，Slurm 的 GRES 还支持一种跟 Nvidia 相关的资源：CUDA Multi-Process Service (MPS)[https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf]。如果对 MPS 尚不了解，我建议先简单看看 MPS 的 example。本质上，MPS 就是允许多个小任务同时运行在一块 GPU 上，并且前一篇文章中提到的 overhead 也大大降低。MPS 通过设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 来限制每个任务的算力。`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 的取值在 `(0,100]` 之间，也就是说，MPS 将一块 GPU 的算力等分成了 100 份。（所以在 Slurm 中，虽然可以配置一块 GPU 的 mps 为 100 的 N 倍，但是依然会被折算成百分比。）

需要注意的是，一块 GPU 不能同时被设定为 `gpu` 和 `mps` 资源。这对调度器是一种不现实的要求。可以通过修改每个节点上的 `gres.conf` 文件来对该节点上的资源进行配置。

用户在给任务配置资源的时候，直接通过配置 `--gres` 为 `mps:50` 来进行 GPU 共享资源的分配。Slurm 调度器在处理 `mps` 资源时，也会在选定节点和 GPU 之后配置 `CUDA_VISIBLE_DEVICES` 来限定 GPU 资源的使用。

可见，Slurm 对 GPU 共享的调度已经做到相当原生。而如果有进一步的需求，也可以通过 (Slurm Plugin API)[https://slurm.schedmd.com/gres_plugins.html] 自己来实现一个。

## Kubernetes & AliyunContainerService

Kubernetes 本身对 GPU 是支持非共享调度的。主要依靠的就是 Device Plugin。如果对 k8s 如何利用 device plugin 进行 GPU 调度流程尚不熟悉的，可以先去看看其他的文章。这里就不赘述了。

可以知道，想要突破 k8s 上对 GPU 的调度限制，有两点必须要做：

1. Node 上必须可以将一块 GPU 多次绑定到不同的容器
2. Scheduler 必须处理非 `nvidia.com/gpu` 这一类的资源

与 Slurm + MPS 按照算力分割略有不同的是，阿里的方案以显存为分割尺度，并且默认地认为 GPU 算力的需求和显存的需求是成正比的。这也是有一定合理性的。

那么先看看阿里是怎么处理第一类问题的。拉起 container 过程由 kubelet 完成，节点上的 device-plugin 只提供节点上加速器（即 GPU）的状态。这个时候可以选择修改 kubelet。然而修改 kubelet，会使得方案很难在其他的 k8s 集群上部署。所以阿里提供了新的 device-plugin。它将用以共享的 GPU 当作 `Extended Resource` 注册，并且统计节点上所有 GPU 显存的总和。当 kubelet 调用 device-plugin 的 `allocate` API 时，device-plugin 会先通过 k8s API 获取所有被调度到该节点但尚未被处理的 GPU Sharing Pod，而后选择老的 Pod（等待时间最久），为其在环境变量中配置从 annotation 获取的 Device ID 给 `CUDA_VISIBLE_DEVICES` 以便实现 GPU 与 GPU 之间的隔离，最后标记为 `assigned`。

而在调度器一层，阿里的方案使用的工具是 Extended Scheduler。当前用以共享的 GPU 已经被注册为一种新的资源 Extended Resource，而一旦 Pod 被调度到。现在需要做的就是让 k8s 的调度器可以正确处理相关的 Pod。由于默认调度器仅能判断一个节点上的 GPU 显存是否足够容纳当前的 Pod，因此 GPU Share Scheduler Extender 会帮助其在做一次过滤，将那些单个 GPU 不足以容纳下显存申请的节点过滤掉。在选择好节点之后，binding 的过程也交由 Scheduler Extender 完成。当中主要的工作是在选择好的节点中，以避免资源浪费的形式选择合适的 GPU、将选择好的 GPU ID 写入 Pod 的 annotation、将 Pod 与 Node 绑定。

## What's next?

那么将多个任务调度到同一个 GPU 之后，我们如何去限制算力和显存这两项 GPU 资源呢？下一篇文章，会从“力所能及”的尺度介绍腾讯 TKEStack 开源的 GaiaGPU 方案以及 VirtAI 的社区版。大家可能需要先做一些 API 劫持的理论准备哦。

