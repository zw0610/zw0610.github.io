# 在什么条件下，FC == Conv？

*这是一篇读 code 笔记，不涉及任何高深的机器学习理论*

## 用 GEMM 实现 Convolution 操作

*（我们这里提到的 convolution 指的是大部分 dl framework 中的卷积操作，硬杠的话我知道其实这是 correlation）*

cuDNN 中有用 GEMM 来实现卷积操作的。不多说了，文章在[这里](https://arxiv.org/pdf/1410.0759.pdf)。

## Apple MPS

Apple 也有一个类似于 cuDNN 的东西，叫 Metal Performance Shaders。Metal 可以认为是苹果自己的 OpenCL。

我在看 Apple 提供的[如何利用 Metal 和 MPS 训练一个简单的卷积网络](https://developer.apple.com/documentation/metalperformanceshaders/training_a_neural_network_with_metal_performance_shaders?language=objc)的时候，发现没有任何 **Flatten** 操作。这个 Flatten 操作指的是在做完卷积层后，把结果的 2D（单通道） 或者 3D（多通道）的一个张量延展成一个向量的操作。`PyTorch` 里面一般就是用 [`view(-1, num_flat_features)`](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) 这个操作来实现。

但是 Apple 的例子中完全找不到这个操作：

```Objective-C
    // conv layer
    MPSCNNConvolutionNode *conv2Node = [MPSCNNConvolutionNode nodeWithSource:pool1.resultImage
                                                                     weights:conv2Wts];
    conv2Node.paddingPolicy = sameConvPadding;
    // element-wise activation
    MPSCNNNeuronReLUNode *relu2 = [MPSCNNNeuronReLUNode nodeWithSource:conv2Node.resultImage a:0.f];
    // pooling
    MPSCNNPoolingMaxNode *pool2 = [MPSCNNPoolingMaxNode nodeWithSource:relu2.resultImage filterSize:2 stride:2];
    pool2.paddingPolicy = samePoolingPadding;
    // linear/fully-connected
    MPSCNNFullyConnectedNode *fc1Node = [MPSCNNFullyConnectedNode nodeWithSource:pool2.resultImage
                                                                         weights:fc1Wts];
    // element-wise activation
    MPSCNNNeuronReLUNode *relu3 = [MPSCNNNeuronReLUNode nodeWithSource:fc1Node.resultImage a:0.f];
       
    MPSNNFilterNode *f2InputNode = relu3;
    // another linear/fully-connected
    MPSCNNFullyConnectedNode *fc2Node = [MPSCNNFullyConnectedNode nodeWithSource:f2InputNode.resultImage
                                                                         weights:fc2Wts];
```

可以看到，在 Padding 之后，MPS 的 model 直接接了 FC 层。是 MPS 中的 FC 自动实现了 Flatten 的操作吗？虽然 Apple MPS 的 doc 确实很差劲，但是我也自己找了，确实没有找到相应的说明。那我们不妨就来比较一下 `fc1Node` 和 `fc2Node` 两层的 weights 的区别吧：

```Objective-C
    fc1Wts = [[ConvDataSource alloc]  initWithKernelWidth:7
                                             kernelHeight:7
                                     inputFeatureChannels:64
                                    outputFeatureChannels:1024
                                                   stride:1
                                                    label:@"fc1"];

    fc2Wts = [[ConvDataSource alloc] initWithKernelWidth:1
                                            kernelHeight:1
                                    inputFeatureChannels:1024
                                   outputFeatureChannels:10
                                                  stride:1
                                                   label:@"fc2"];
```

这里有部分“猜”的成分，希望大家谅解。我们先看到 `fc2Wts`，这是最后一个 FC 层的 weights。显然 `inputFeatureChannels` 是该 FC 层的 input features。而 `outputFeatureChannels` 也跟这是个十分类（MNIST 数据集）的问题相吻合。接下去就是接 Softmax 层了。

那么确定了这些信息之后，我们再回头看 `fc1Wts`。`outputFeatureChannels` 是 1024 没有问题，和下一层对得上。而 `inputFeatureChannels` 只有 64，这就很奇怪了。但是我们要注意的是，对于 `fc1Wts`，其 `Width` 和 `Height` 都不是 1，而是 7。***这就意味着对于前面做完 pooling 得到了一个 channel 为 64 的 7x7 张量，而对其做一个特殊卷积得到的结果其实就是展平后做一个向量乘矩阵。*** *（<-这个结论我还没实打实算过到最后做完 pooling 是不是 7x7）*。

这听上去似乎是挺合情合理的做法，尤其是在看了上面提到的 NVIDIA 的那篇文章之后。

## 数学上证明

最后我们需要在数学上证明对于一个 7x7x64 的张量，用一个 7x7x64x1024 的 kernel 做 correlation 得到的结果，等价于将该张量展平并将这个 kernel 进行变换后进行的一次向量乘矩阵。

我们这里都假设矩阵采用 row-major 吧。也就是说一个 3x3 的矩阵可以这样转换成向量：

| | | |
|-|-|-|
| a_1_1 = vec_0 | a_1_2 = vec_1 | a_1_3 = vec_2 |
| a_2_1 = vec_3 | a_2_2 = vec_4 | a_2_3 = vec_5 |
| a_3_1 = vec_6 | a_3_2 = vec_7 | a_3_3 = vec_8 |

- 我们假设 input channel 是 1，output channel 也是 1。这样问题退化成对一个 7x7 的矩阵，用一个 7x7 的 kernel 做 correlation 得到的结果。最后得到的应该是一个长度为 1 的向量。

按照我们约定的张量转换成向量的方式，那么这个 7x7 的 kernel （张量）转换成矩阵其实就是先按照同样的方式把 kernel 张量转成一个向量，然后以这个向量为一个矩阵的第一列，把向量里的元素从上到下进行放置。（当然其实就是对这个向量做 transpose，得到是一个竖向量。）

- 我们假设 input channel 不是 1，而是大于 1 的 P。这时候首先我们再定义从张量转换到向量的规则：channel first，也就相当于现针对每个 channel 的 7x7 张量单独做转换，然后按照 channel 的顺序把 P 个子向量拼接成整体向量：

``` [a_1_1_channel1, ... , a_7_7_channel1, a_1_1_channel2, ... , a_7_7_channelP] ```

这个时候，我们的 kernel 自然也变成了 Px7x7 的。其转换方法不需要改变，得到一个 49x1 的矩阵/竖向量。

- 我们假设 output channel 不是 1，而是大于 1 的 Q。那么最终结果应该是一个长度为 Q 的向量。

kernel 张量的变化其实也很简单。对于 Q 中任一 channel，先按照 2 中的方法转成向量，然后根据其在 Q 中的顺序，以列的形式排列到矩阵中。也就是说在 output channel 中的第 i 个 Px7x7 张量，先变成一个长为 Px7x7 的向量，然后作为第 i 列拼入矩阵中。

这样一来，整个过程就完备了。

饿，所以说这其实是退化成了 NVIDIA 那篇文章的一个特例吧。好吧，真是献丑了。

可是既然如此，为啥其他的 code 里大家都会用一个 `Flatten/view(-1)` 然后再用一个 linear op 呢。。。 