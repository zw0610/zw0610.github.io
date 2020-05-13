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

## RMSProp

TBD

## Adam

## Adadelta

## Adagrad