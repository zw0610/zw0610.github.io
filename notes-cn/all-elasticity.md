# 也说弹性分布式训练

本人不才，写了一个颇为简陋的弹性分布式训练工具：[FTLib](github.com/caicloud/ftlib)。所以今天，我也来谈一谈弹性伸缩的分布式训练。

## 容错和弹性

实话实说，一开始只是想做一个容错的功能。那什么是分布式训练的容错呢？

一开始源自于面试时的一个问题：如果一个分布式训练的其中一个 worker 失效了，如果要将这个分布式训练继续？

其实最简单的办法就是每个 epoch 保存好 checkpoint，一旦失效就开启一个新的分布式训练任务，从上一个 checkpoint 读取先前的状态然后继续训练。

这么优秀的方法，为什么大家都不喜欢用呢？或许是要求算法工程师在写脚本的时候加上 save checkpoint 比较为难吧（误！）。大家似乎都想有一种分布式训练，在一个或若干个 worker 失效之后，只要有一个 worker 尚且存在，就可以继续训练。这样一来似乎有如下有点：

1. 当前 epoch 的训练结果不会被废弃
2. 可以保证训练继续
3. 等（以后想到了再补充）

期待 1 的理由无非是因为当前训练的数据量可能颇大，一个 epoch 就会耗费几个小时甚至十几个小时。这样一来，每一个 batch 都是珍贵的。
期待 2 的原因在于，对于很多调度器，如果当前任务结束再重新开始，可能剩余资源就无法满足新的任务。猿多卡少的局面往往不是一家公司独有。

那我们假设，就是有了这样一个工具：粉身碎骨浑不怕，单留 worker 在集群。

于是一些人，粥还没吃上，就想吃干饭了：既然可以容错，那可不可以扩容呢？即，针对一个分布式训练，用户多加进去一个或多个 worker(s)，该 worker(s) 可以参与到已经开始的训练中来。

这就是弹性了。它的好处？最大的好处莫过于可以吸纳集群闲置的 GPU 资源。谁让核弹黄把卡卖得那么贵。无论公司大小，总都是希望物尽其用的。

## 仅限数据并行？

我们刚才谈到分布式训练的时候，你会发现并没有特指数据并行的分布式训练还是模型并行的分布式训练。但是你看到随处可见的 *worker* 一词，好像又暗示我们这里只讲数据并行的分布式训练。

没错。模型并行的分布式训练并不在我们这次讨论之列，原因有三：

1. 模型并行往往针对单块 GPU 容纳不下的模型，这样的模型使用者相对少一些
2. TF 本身的 PS/Worker 设计大致可以做到容错
3. ElasticDL 团队先前已经实现了 PS/Worker 的容错和弹性

所以我们这里讨论的是采取 allreduce 来做同步的梯度更新的数据并行分布式训练。

## “百”家争鸣

我在 19 年年中向当时的公司提出了要做弹性/容错的分布式寻训练这个计划。非常有幸的是，公司当时就同意了。这使得我们处于弹性分布式训练的第一批次。其他的方案还包括：

1. torch/elastic(released)
2. horovod(wip)

在设计 FTLib 的初始，我们提出过基于 zookeeper 的方案，即利用一个外置的一致性存储在保证所有的 worker 可以对当前参与本次分布式训练任务的成员有一个统一的认识。这个认识，我们称之为**成员表共识**，即 **consensus of memberlist**。每个成员通过 beat 的方式来不断登记自己的状态。这个方案和 `torch/elastic` 不谋而合。`torch/elastic` 也把每个 worker 需要的成员共识转移到了 etcd （默认）或是其他存储上，并设定了 memberlist 的上限与下限：低于下限不进行训练，高于上限不吸纳新的 worker。

然而这个转移一致性的方案在评审中却被毙了。一方面可能是因为这个方案不够优美，但更重要的我想是因为这个方案需要借助第三方组件的方式并不利于方案的快速部署。在 TL 的启发下，我开始着手基于 Gossip 协议的 FTLib。

Gossip 协议就像病毒传播一样，每个 worker 都会随机的查询/推送内容从/到若干个随机节点中。它提供了一种弱的一致性保证，在一定时间后，所有的 worker 对 memberlist 会有一个一致的认识。考虑到一个分布式训练任务并不会频繁地扩缩 worker，那么即便在更改 memberlist 时稍微耗费一些时间，也是不要紧的。于是，这就成为了 FTLib 的基石。如果你对 FTLib 的原理和设计有兴趣，可以看一下我之前在公众号中写的这篇 [PR 稿](https://www.infoq.cn/article/D8uzbcBAGNJHDzHyA3He)。

那么 horovod 呢？好像还没听说。没错，我也是在刷他们的 [pull request](https://github.com/horovod/horovod/pull/1849) 的时候偶尔发现的，现在应该还在开发中。

horovod 似乎很老实地把 consensus of memberlist 划在了他们的职能之外，而是要求用户提供一个 `discover_host.sh` 脚本，交由 `horovodrun` 组件来不断 check，从而实现弹性与容错。嗯，平平淡淡才是真嘛。我还记得有人评价我写代码过于追求优美，而往往久经考验的代码采用的是最简单的逻辑。

## 跬步

我们在有了一个支持弹性的分布式训练的工具之后，不要说离提高集群的 GPU 利用率进了一大步，就是离真正的弹性训练本身也只是走了一小步。有两个问题一下子暴露了出来：

1. worker 数量变更后，剩余训练数据如何分配？
2. worker 数量变更后，学习率要如何修改？

如果两者不做修改，那么问题是显然易见的。“全量数据”四个大字让这个 epoch 变成了鸡肋，而模型的收敛与否更是关系到这个 epoch 能否真正走完。

尽管有一些初级的探索，例如等比缩放学习率等等，其实践效果依然尚待验证。而这个时候，异步更新的方式似乎更能胜任这种 worker 数量的变化。

回过头来说提高集群的 GPU 利用率，在拥有了弹性分布式训练之后，我们依然需要一个相当智能的任务队列，它需要具备以下几项功能才能真正发挥弹性训练的效果：

1. 每个任务可以设置优先级
2. 当有优先级高的任务来临时，可以自动减少低优先级的分布式训练的 worker 数量
3. 当集群资源闲置时，可以自动增加 worker 数量
4. 分布式训练的 worker 失效后，会被自动拉起
5. 要是有资源超卖功能就更好了

鉴于我属于“失业中”，还是不去多想什么时候能够实现吧。今天就先说到这儿。