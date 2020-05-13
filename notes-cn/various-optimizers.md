# 优化器 All-in-One

今天面试的时候又被问到了关于各种 optimizers 的问题，什么 RMSProp 是什么、跟 Adam、Momentum 的区别在哪里。我又一次没记住，毕竟平时都是做工程的。然后当我结束后去查询的时候，发现中文社区对与 Optimizers 的总结真是一言难尽。秉承这“好记性不如烂笔头”的宗旨，还是把自己看文章的收获记下来，以备下次面试时会用到。

## Baseline SGD with Momentum

鉴于很多深度学习框架直接把带 Momentum 的 SGD 当作 default SGD optimizer，这里就把他们合在一个章节里，作为 Baseline。



### Mini-Batch SGD

Gradient Descent 的核心思想非常的简单。针对一个模型，我们最终希望它的（非负的）损失函数最小：`𝜽 = argmin J(𝜽)`。

根据 [Gradient Descent on Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)，GD 秉持一个最朴素的思想：`J'(𝜽)` 指向哪里，沿着它的反方向 `-J'(𝜽)` 走（即 `𝜽_next = 𝜽_now - J'(𝜽)`），就能让 `J(𝜽)` 降低。这个思想甚至比高斯迭代还要来得更朴素。

然而一次性计算做全样本的梯度计算是非常昂贵的，计算资源不说，显存的占用恐怕就让 GPU 望而却步。所以一般都采取 Mini-Batch 的方法，即把全量数据拆分成 N 个 Batches，然后每过一个 Batch，就更新一下 `𝜽`。相交 SGD，确实没有那么昂贵了，但是从收敛稳定的角度讲，不如全量数据的 SGD。原因在于，全量数据 SGD 在更新梯度时，可以站在全局数据给出的角度计算出最为精确的梯度；而在 Mini-Batch 引入之后，每一次梯度的更新都是相对于全量数据的局部梯度，颇有（半）盲人摸象的感觉。

### Momentum & Nesterov Momentum

刚刚提到了 Mini-Batch 的一个缺陷，就是每次更新 𝜽 的时候，由于仅立足与一个小批样本，如果这小批样本和全局样本的分布有较大的差异的话，那么这一次梯度更新就会较大程度上地偏离最终的 𝜽_optimal。事实上，每一个 Mini-Batch 其实都会造成梯度更新的偏离。

那如何解决呢？有一个办法跟 Gauss–Seidel 方法衍生出来的 Successive Over-Relaxation (SOR) 颇有类似。简而言之，就是*既然这一次更新可能会走偏，那我就不要全速更新，在一定程度上保有原有的值*（颇有[两个基本点](https://zh.wikipedia.org/wiki/%E4%B8%80%E4%B8%AA%E4%B8%AD%E5%BF%83%E3%80%81%E4%B8%A4%E4%B8%AA%E5%9F%BA%E6%9C%AC%E7%82%B9)的意思）。

Momentum 在 DL 上的的具体方法我查阅了 [On the importance of initialization and momentum in deep learning](http://www.cs.utoronto.ca/~ilya/pubs/2013/1051_2.pdf)。

最为原始的 Momentum 方法即为利用上一次更新的方向做一次“纠偏”：

```
v_t+1 = 𝝁v_t - 𝜺J'(𝜽_t)
𝜽_t+1 = 𝜽_t + v_t+1
```

其中，`𝜺` 是学习率，`𝝁` 则为一个取值 `[0,1]` 的动量系数。

接下去会有一种改进版的 Momentum 法：Nesterov's Momentum。

```
v_t+1 = 𝝁v_t - 𝜺J'(𝜽_t + 𝝁v_t)
𝜽_t+1 = 𝜽_t + v_t+1
```

通过对比，可以很明显地知道，Nesterov 法其实就是在计算梯度时， 并不是从当前 `𝜽_t` 出发进行计算，而是假设走一步 `𝝁v_t` 之后的中间值。这种提前走半步的方法可以在一定条件下加速模型参数的收敛。

## 众口难调？

在解决了 Mini-Batch 带来的抖动问题之后，还有一个问题困扰着 DL 模型的优化，即多参数带来的问题。所谓多参数，就是说我们上面看到的 `𝜽` 其实可能是一个长度超过百万的向量。尤其是随着深度学习模型的复杂化，参数的数量也在膨胀。可以想像，用一个单一的学习率作用于所有的参数，其带来的梯度更新的效果必然会收到比较大的影响（有的参数已经比较接近 optimal 了，而有的还差很远）。针对这一问题，又有哪几种解决办法呢？

### RMSProp
Root Mean Square Propagation（RMSProp）这种方法最开始从 Cousera 上的 [slice](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) 起家，很快收到大家的欢迎。其核心的思想就是针对每一个参数，都利用其之前几次的梯度更新来调整这次更新需要怎样的学习率。

如果不考虑具体的梯度值，只考虑梯度的符号（即方向），RProp 有这样一条规律：

1. 如果过去两次更新的梯度符号相同，代表方向正确，增大更新时的步长
2. 如果过去两次更新的梯度符号相反，代表方向不定，缩小更新时的步长

然而这种方法在 Mini-Batch 上却不好。每个 Mini-Batch 之间的差别会使得每一次更新本身的步长就会有很大的差异，这一点和全量数据更新时有比价明显的差异。这样一来，单纯依靠符号就显得力不从心了。我们需要加上依赖数值的估计来弥合 Mini-Batch 带来的差异。

我们先来计算已经累积的梯度的均方差，这里引入一个遗忘（或阻尼）系数 `𝛾`，使得很久之前的梯度方差不至于过多影响当前的梯度评估。

```
v(w, t+1) = 𝛾v(w, t) + (1-𝛾)(J_w'(𝜽))^2
```

`v(w, t+1)` 估计了过去一段时间内，该参数（`w`）的梯度量级。显然，它是一个正数（初始值>0的话）。

最后，在更新梯度的时候，将步长除以 `v` 的开方：

```
𝜽_w,t+1 = 𝜽_w,t - 𝜂*J_w'(𝜽)/sqrt(v(w, t+1))
```

### Adam

Adaptive Moment Estimation（Adam）优化器针对 RMSProp 做了进一步的优化，这里参考了出处[ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980.pdf)。 

和 RMSProp 相比，Adam 主要的不同指出在于对于更新梯度时，对于梯度 `J'(𝜽)` 加入了阻尼：

```
m_t = 𝜷_1*m_t-1 + (1 - 𝜷_1)*J'(𝜽)
```

然后依然像 RMSProp 那样利用梯度方差：

```
v_t = 𝜷_2*v_t-1 + (1 - 𝜷_2)*J'(𝜽)^2
```

当然，这里也对期望和方差都做了修正 `1/(1 - 𝜷_1^t)` 以及 `1/(1 - 𝜷_2^t)`。

```
m_hat_t = m_t / (1 - 𝜷_1^t)
v_hat_t = v_t / (1 - 𝜷_2^t)
```

最后，继续依旧常规更新梯度。

```
𝜽_w,t = 𝜽_w,t-1 - 𝜂*m_hat_t/(sqrt(v_hat_t) + 𝜺)
```

`𝜺` 是一个用以保证数值稳定的一个很小的值。

### AdaGrad

其实 [AdaGrad](https://web.archive.org/web/20150330033637/http://seed.ucsd.edu/mediawiki/images/6/6a/Adagrad.pdf) 相比 RMSProp 可能更早，可以看作是一个未做阻尼的 RMSProp 版本。它更新参数的方式于 RMSProp 非常相似：

```
𝜽_w,t = 𝜽_w,t-1 - 𝜂*J_w'(𝜽)/sqrt(G_w,w)
```

那这里的 `G_w,w` 是怎么计算出来的呢？`G_w,w = Sum[J_w_t'(𝜽)^2, {𝝉, 1, t}]`，这里 `𝝉` 指的是迭代的次数，换句话讲，就是记录下了该参数的历史梯度平方和。

### Adadelta

AdaDelta 的文章，[ADADELTA: AN ADAPTIVE LEARNING RATE METHOD](https://arxiv.org/pdf/1212.5701.pdf)，相当有意思。

文章先介绍了一种 idea：Accumulate Over Window。

文章后面还提到了第二种想法，利用二阶导数 Hessian 矩阵进行梯度下降的纠正。这里就暂时不展开了。关于利用 Hessian 矩阵的优化，我们有空在另一篇文章中来看。

Accumulate Over Window 这种方法似乎是介于 AdaGrad 和 Adam 之间的一种方法。AdaGrad 中直接累加过去所有的梯度平方。但是显然，越久之前的梯度平方越难以反映临近当前的梯度状况。因此我们需要对梯度平方做一个加权：

```
E_t[g^2] = 𝝆E_t-1[g^2] + (1-𝝆)g_t^2
```

同样的，对梯度下降的速度也做一个估计，用来当作学习率：

```
E_t[𝜟x^2] = 𝝆E_t-1[𝜟x^2] + (1-𝝆)𝜟x_t^2^2
```

然后同时用这两项来更新梯度：

```
x_t+1 = x_t - sqrt(E_t-1[𝜟x^2])/sqrt(E_t[g^2])*J'(𝜽)
```


*P.S. 等我掌握如何用 gist 渲染数学公式之后，我会把这里的表达式再更新一篇*
