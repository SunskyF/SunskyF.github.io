---
title: cs231n spring 2017 assignment1
date: 2018-01-18
category:
    - 公开课
tag:
    - 深度学习
---

做完这门课的作业也蛮久了，但好像每次自己想的时候都回忆不起来，还是写下来有助于记忆，和便于下次再查看。学一个东西，还是得需要更深入的去理解它，如果只是程序填空或者跑别人的代码，那即使你现在看懂了，但实际上还是没有进入大脑。

[代码](https://github.com/SunskyF/cs231n_spring_2017/tree/master/spring1617_assignment1/assignment1) 这里提供了下文中的代码。

## Q1: k-Nearest Neighbor classifier (20 points)

knn可以说是很基础的分类模型，想法很简单：当我们想要给一张图片进行分类的时候，从我们的训练集中找到k张最接近的图片，然后根据它们进行投票，得出最后的分类结果。

<!-- more -->

我们需要完成的部分是：

### compute_distance

比如说我们有Ntr个训练样本和Nte个测试样本，我们需要得到Nte * Ntr的矩阵，里面的每一行对应的是测试样本和所有训练样本的距离。

#### compute_distances_two_loops

最朴素直接的想法，算一下两个样本之间的l2距离即可。

#### compute_distances_one_loop

利用numpy的广播机制

```
dists[i, :] = np.linalg.norm(self.X_train - X[i], axis=1)
```

再针对对应的那个维度做一下norm即可，与上面的改变就是我们直接计算了一个测试样本到所有训练样本的距离。这也就是向量化操作，因为矩阵的操作远比显式的循环快。

#### compute_distances_no_loops

这个就有点trick了。

$(x - y)^2 = x^2 - 2xy + y^2$

其实用的是这个很普通的公式。我们计算l2距离的时候，其实就是在算$(x-y)^2$，我们把这个式子展开，就可以避免compute_distances_one_loop中的循环。

```
xy = X.dot(self.X_train.T)    
xx = np.sum(X ** 2, axis=1, keepdims=True)
yy = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
dists = np.sqrt(-2 * xy + xx + yy)
```

经过测试，我们发现速度上的差异还是很明显的。

```
Two loop version took 29.392334 seconds
One loop version took 45.736492 seconds
No loop version took 0.253548 seconds
```

### Cross-validation

模型中往往有很多超参数，比如knn里的k。我们不能直接使用测试集来进行调参，因为这样我们其实是将测试集用成了训练集。所以我们往往会从训练集中分一部分出来用于评估模型，这部分叫做验证集（validation set）。

而cv就更复杂一点，如果5-fold，我们将训练数据均分为5份，然后取4份训练，剩下1份作为验证集，然后换4份训练，最后平均一下。

## Q2: Training a Support Vector Machine (25 points)

支持向量机是一个很经典的算法，有着很强的数学基础，非常优雅。

[课程note](http://cs231n.github.io/linear-classify/)上提供了loss函数，$L_i = \sum \limits_{j\neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)$

我们对w求导
$$
\frac{dL_i}{dw_j} =
\left\{
\begin{aligned}
x_i, j \ne i \\
-\sum \limits_{i \ne j} x_i , j =i
\end{aligned}
\right.
$$
这里有我对svm的简单学习笔记{% post_link svm-intro %}。

我们将
$$
\min \limits_{w, b}\frac{1}{2}{||\textbf{w}||}^2 + C\sum \limits_{i=1}^m l_{0/1}(y_i(w^Tx_i+b) - 1), C > 0
$$
拿出来看，我们使用hingle loss替换01损失函数

我们得到
$$
\min \limits_{w, b}\frac{1}{2}{||\textbf{w}||}^2 + C\sum \limits_{i=1}^m\max(0, 1- y_i(w^Tx_i+b)), C > 0
$$
我们回想一下Linear Regression的loss函数的样子
$$
min \sum \limits_{i=1}^m(y_i - w^Tx_i)^2 + \frac{1}{2}{||\textbf{w}||}^2
$$
前一项是loss，后一项是正则项，这不就和上面svm的式子很像嘛？将其扩展为多分类问题之后，就是课程所提供的loss函数了。

### SVM Classifier

#### svm_loss_naive

```python
for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]] # s_{y_i}
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      # s_{j} - s_{y_i} + delta
      margin = scores[j] - correct_class_score + 1 # delta = 1
      if margin > 0:
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
        loss += margin
```

#### svm_loss_vectorized

```python
num_train = X.shape[0]
score = X.dot(W)
correct_class_score = score[np.arange(num_train), y] # 每一个样本的正确类别的得分
margin = score - correct_class_score[..., None] + 1 # s_{j} - s_{y_i} + delta
margin[np.arange(num_train), y] = 0 # 正确的类别就置0，便于后续处理
loss += np.sum(margin[margin > 0]) # max(0, ...)
loss /= num_train
loss += reg * np.sum(W * W)

coe = np.zeros(margin.shape)
coe[margin > 0] = 1 # 是要计算grad的
coe[np.arange(num_train), y] = 0
coe[np.arange(num_train), y] = -np.sum(coe, axis=1) # 系数
dW = X.T.dot(coe) # 直接求了累积
dW /= num_train 
dW += 2 * reg * W
```

#### LinearClassifier.train

```python
batch_idx = np.random.choice(num_train, batch_size) # 随机选择一个batch
X_batch = X[batch_idx]
y_batch = y[batch_idx]

self.W -= learning_rate * grad # 结合学习率更新参数
```

#### LinearSVM.predict

```python
y_pred = np.argmax(X.dot(self.W), axis=1) # 选择输出得分最高的作为预测结果
```

## Q3: Implement a Softmax classifier (20 points)

$$
softmax: f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

对这个求导
$$
\frac{df_j}{dz_i} = \left\{
\begin{aligned}
\frac{e^{z_j}(\sum_k e^{z_k} - e^{z_j})}{(\sum_k e^{z_k})^2} = f_j(z)(1-f_j(z)), j = i \\
-\frac{e^{z_j}e^{z_i}}{(\sum_k e^{z_k})^2}  =- f_j(z)f_i(z), j \ne i
\end{aligned}
\right.
$$

下面我们结合*cross-entropy loss*
$$
L_i = -log(\frac{e^{f_{y_i}}}{\sum_j e^{j}})
$$
求导
$$
\frac{dL_i}{dz_j} = \left\{
\begin{aligned}
-\frac{1}{f_i(z)}\frac{e^{z_j}(\sum_k e^{z_k} - e^{z_j})}{(\sum_k e^{z_k})^2} = f_j(z) - 1, j = i \\
-\frac{1}{f_i(z)} * -\frac{e^{z_j}e^{z_i}}{(\sum_k e^{z_k})^2}  = f_j(z), j \ne i
\end{aligned}
\right.
$$


### Softmax Classifier

#### softmax_loss_naive

```python
for i in range(X.shape[0]):
    h = X[i].dot(W)
    pred = np.exp(h - h.max()) / np.sum(np.exp(h - h.max()))
    loss += -np.log(pred[y[i]]) + 0.5 * reg * np.sum(W ** 2)
    pred[y[i]] -= 1
    dW += X[i][..., None].dot(pred[None, ...]) + reg * W
```

#### softmax_loss_vectorized

```python
h = X.dot(W)
pred = np.exp(h - h.max(axis=1, keepdims=True)) / np.sum(np.exp(h - h.max(axis=1, 				keepdims=True)), axis=1, keepdims=True)
loss += np.sum(-np.log(pred[np.arange(X.shape[0]), y])) / X.shape[0] + 0.5 * reg * np.sum(W ** 		2)
pred[np.arange(X.shape[0]), y] -= 1
dW += X.T.dot(pred) / X.shape[0] + reg * W
```

## Q4: Two-Layer Neural Network (25 points)

前向非常简单，直接叠加起来就行
$$
h = relu(w_1^Tx + b_1) \\
y = w_2^Th + b_2 \\
z = softmax(y) \\
loss = -log(z)
$$
反向也很简单，我们在Q3中已经推导了$\frac{dL_i}{dz_j}$，也就是这里的$\frac{dloss}{dy}$。

那么我们可以用链式法则
$$
\frac{dloss}{dw_2} = \frac{dloss}{dy}\frac{dy}{dw_2} = h^T\frac{dloss}{dy} \\
\frac{dloss}{db_2} = \frac{dloss}{dy}\frac{dy}{db_2} = \frac{dloss}{dy} \\
\frac{dloss}{dw_1} = \frac{dloss}{dh}\frac{dh}{dw_1} = \frac{dloss}{dy} \frac{dy}{dh}\frac{dh}{dw_1}
$$
然后按照推出来的公式写代码就好了，注意一下矩阵的维度就能理清$w$写在前面还是后面。

####  forward pass

```python
h1 = X.dot(W1) + b1 # w^Tx + b，第一层
relu1 = np.maximum(0, h1) # max(0, x)，激活函数relu
scores = relu1.dot(W2) + b2 # w^Tx + b，第二层

pred = np.exp(scores - scores.max(axis=1, keepdims=True)) / np.sum(np.exp(scores - 					scores.max(axis=1, keepdims=True)), axis=1, keepdims=True) # 同softmax
loss = np.sum(-np.log(pred[np.arange(X.shape[0]), y])) / X.shape[0] + reg * (np.sum(W1 ** 2) + 			np.sum(W2 ** 2))
```

#### backward pass

```python
dscores = pred 
dscores[np.arange(X.shape[0]), y] -= 1 # softmax的梯度
grads['W2'] = relu1.T.dot(dscores) / X.shape[0] + 2 * reg * W2
grads['b2'] = np.sum(dscores, axis=0) / X.shape[0]
dh1 = dscores.dot(W2.T)
dh1[h1 < 0] = 0
grads['W1'] = X.T.dot(dh1) / X.shape[0] + 2 * reg * W1
grads['b1'] = np.sum(dh1, axis=0) / X.shape[0]
```

#### tune hyperparameters

loss曲线差不多以线性的方式递减，说明学习率设置的太小，loss经过一定次数的迭代，还没有收敛。

训练集和验证集之间的准确度曲线差距不大，说明模型的容量不够大。我们对一个容量足够大的模型会希望它能够overfitting，也就是训练集和验证集之间会有比较大的差距。

也就是在这里，我们可以调整的参数是隐藏层的大小、学习率、迭代册数、正则项参数。

## Q5: Higher Level Representations: Image Features (10 points)

主要就是想告诉我们，经过提取特征训练的分类器比使用原始数据训练的好。

在python3下，可能运行有问题，主要是索引不能为float的锅。

把feature.py中121行改为

```python
orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx//2::cx, cy//2::cy].T
```

第二行加一下

```python
from __future__ import division
```

## 后记

过了很久，终于把这个写好了，感觉自己对这些知识的推导更加清晰了。

这也是我第一次认真的写博客，希望之后能够坚持吧。