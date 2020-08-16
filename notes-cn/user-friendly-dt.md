# 分布式训练中对"用户友好"的一点点思考

> *这里不会涉及到高深的技术性思考或者底层代码，纯粹是从产品设计的角度来思考分布式训练。*

## 有面向算法工程师的产品设计吗？

我身边经常有产品设计同学，或者叫产品经理。相信很多研发同学对他们又爱又恨([呼兰在云栖大会上的段子](https://www.bilibili.com/video/av69095180/))。恨自然是不用多讲了，那么爱呢？一个优秀的产品经理是可以洞察用户的真正需求，设计出直达用户内心的产品的。有 ToB 的产品经理，也有 ToC 的产品经理。

然而当我们把领域限缩到深度学习产品/工具时，似乎很少见到面向算法工程师的 **专业的** 产品经理。这当然主要还是来自于前几任公司的感受。我毕竟还是新鹅，对鹅厂的了解还非常初步。在涉及到深度学习产品的时候，无论是训练框架、推理框架、平台还是其他的一些深度学习工具，很多开发者都采取了一种“工程师文化”的方式，实践着“工程师即产品设计”、“技术驱动”等理念。它的优劣无意去评判，但是有一点相信大家很快会发现，即有的时候，设计开发产品的工程师和使用的工程师不是同一类型的工程师。*(特别注明：这里没有反对或贬低工程师文化。)*

这里要说的例子，就是分布式训练。在设计开发分布式训练模块的时候，设计开发人员的背景往往是具有集群管理、资源调度、分布式计算、以及部分的深度学习计算。而使用者的技能树一般都是点在了深度学习算法、数值计算、某一块领域的应用（广告推荐、图像识别等等）。这两者之间的 gap 如果真的了解，其实是比较大的。这也导致有的框架的分布式计算框架学习成本颇高，需要算法工程师掌握相当多的集群知识才能够单独使用。长期以来，大家似乎都在探索怎么样用最小的努力，把一个单机的训练代码变成分布式训练的代码。

## 各个主要的框架怎么处理分布式训练？

为了让讨论的范围明晰，我们这里只谈论数据并行的分布式训练。

先来看看主流的两个吧：TensorFlow 和 PyTorch。

### PyTorch & DDP

PyTorch 对数据并行的分布式训练很好理解，核心在于 DDP（[Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)）。站在算法工程师的角度，就是我原先写了一个 `nn.Module` 可以用作单机训练；如果数据太多要跑分布式，我就把它丢进一个叫 `DDP` 的 wrapper 里去，出来的那个新的 ddp Moodule 依然具备 `nn.Module` 的各项属性，依旧按照单机训练时的操作进行。

然而真的就这样 OK 了吗？我们参考 [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) 的例子就不难发现，其实还有一个 `process_group` 的概念不可避免地引入。需要首先对 `process_group` 进行初始化时，算法工程师可能对 `rank` 这个概念比较陌生。但是用过 MPI 的人，也就是开发分布式训练的工程师一定都非常熟悉。

### TensorFlow & Distributed Strategy

我个人对 TensorFlow 1.x 的分布式训练并不是很喜欢。从早年的 PS/Worker 模式起，就留下了“TF 搞分布式很麻烦”的刻板印象。这种暴露相当多原生概念和接口（如`ClusterSpec`、`task_index`、`job_index` 等等）当然是方便各种定制化操作。但是恐怕 TF 的工程师没有考虑到我们大部分人的技能光谱和他们并不重合。

或许是收到了这方面的反馈，TF 后续推出了 [Distributed Strategy](https://www.tensorflow.org/guide/distributed_training) 这个概念，自然是想减少算法工程师使用 TF 进行分布式训练的学习难度。从[训练代码](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#train_the_model_with_multiworkermirroredstrategy)看，尽管依然需要对 `TF_CONFIG` 进行[合乎规则的配置](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/distributed_training.ipynb#TF_CONFIG)，使用难度确实是降低了。

相较于 PyTorch 的 `process_group`，相同的概念应该是对机器的发现，主要指参与训练机器的 IP 和端口。`Rank` 这个概念在 `MultiWorkerMirroredStrategy` 中变成了 `index`。同时为了使得 `TF_CONFIG` 的格式可以兼容各种 Strategy，用户也不得不了解 `worker`、`parameter server` 这种概念。

## OneFlow 的分布式

实话实说，我是想蹭个热度。这不是 OneFlow （以下用 of 代指）刚开源嘛，所以就急匆匆地看了看如何在 of 上搞分布式，毕竟“低成本从单机到分布式”是 of 的卖点之一。

从[代码](https://docs.oneflow.org/code/basics_topics/distributed_train.py)层面看，of 对于单机到多机的过度介于 Distributed Strategy 和 Distributed Data Parallel 之间。

对机器的发现基本上都是一致的，of 需要用 `oneflow.env.machine(nodes)` 来显式地定义。

而将单机模型变成并行模型时，of 相交 TF 2 更进了一步。在 TF 2 中，还需要利用 `strategy.scope` 来为多机计算的 ops 提供一个 context/scope，单机和多机需要用到不同的 Strategy。而 of 直接将这种 context/scope 隐藏进了 `env` 中。所以观察代码，对于训练的定义，单机和多机并无差异，都只需要为 training function 加上同一个修饰器：`@flow.global_function(type="train")`。个人认为这是做的比较好的地方。

*（既然说了不涉及底层代码和设计，这里就不提 of 是怎么实现的。有兴趣的朋友可以参阅相关的 [Post](https://cloud.tencent.com/developer/article/1675443)。文章中提到 **Constant View** 是 OneFlow 的一大特色。在 Data Parallel 和 Model Parallel 切换的时候，PyTorch 和 TF 是相对比较生硬的。而 OneFlow 的做法显然更用户友好。但这不是本文重点，以后有机会再讨论在这两种 parallelism 之间的切换。）*

## 一点思考与期许

应该很多人会觉得这里讲的东西非常的细枝末节，有点小题大做吧。

做分布式训练，大家的着眼点更多是放在了如何提高计算和通信效率、如何增加两者之间的隐藏、如果避免加速比下滑、如何支持更大规模的模型等等。“用户友好”这一点似乎排名相当靠后。

也有可能是因为之前“多年”跟甲方打交道的经历吧，如何把一个产品作到让用户开心地用起来在我看来还是挺重要的。毕竟不想让工作成果停留在 KPI 上。而存在于更多同事和业界的代码上的工作成果，应该是一种更大的满足感吧。