# 5 Machine Learning Basics

`page 96`

> In this relatively formal definition of the word "task", the process of learning itself is not the task.

The process of learning becomes the "task" in the context of meta-learning, or learning to learn. Without going into too much detail, the rough aim of meta-learning is to train an algorithm that can rapidly adapt to/pick up a new task. A commonly-used benchmark is few-shot classification, where the trained algorithm is given as few as one sample per class for several new unseen classes and then has to predict the class for a new sample. Papers include: ([Vinyals et al., 2016](https://arxiv.org/abs/1606.04080)) ([Snell et al., 2017](https://arxiv.org/abs/1703.05175)) ([Ravi & Larochelle, 2017](https://openreview.net/forum?id=rJY0-Kcll)) ([Finn et al., 2017](https://arxiv.org/abs/1703.03400))

`page 96`

> Machine learning tasks are usually described in terms of how the machine learning system should process an example.

As mentioned in chapter 1, the representation of sample has a huge impact on the ease and success of training a learning algorithm.

> For example, the features of an image are usually the values of the pixels in the image.

It is straightforward to directly represent a digital image with the pixel values ie. a 28x28 image has 784 pixels and can be represented as a 784-dimensional vector. It can be more effective to first normalize the values of the image, since typical RGB images have pixel values of datatype `uint8` (the possible values are integers spanning 0 to 255). 

There has also been some work done that encode the pixel values as one-hot vectors, where each channel value is represented with a 256-dimensional vector ([van den Oord et al., 2016a](https://arxiv.org/abs/1601.06759)) ([van den Oord et al., 2016b](https://arxiv.org/abs/1606.05328)) ([Parmar et al., 2018](https://arxiv.org/abs/1802.05751)).

So in some sense, the representation of the input is rather task-specific and requires a bit of consideration.

`page 98` **Machine Translation.** As of writing this, the current SOTA for machine translation is by Vaswani et al. ([2017](https://arxiv.org/abs/1706.03762)), where they used a purely attentional (no CNN or RNN) architecture called the Transformer network. I have previously done some [notes](https://github.com/greentfrapp/deeplearning-papernotes/blob/master/notes/transformer.md) and [implementation](https://github.com/greentfrapp/attention-primer).

`page 98` On a marginally related note to Structured output, Redmond & Farhadi's ([2018](https://arxiv.org/abs/1804.02767)) YOLOv3 paper (tackling object detection) is the funniest research paper I've had the pleasure to come across.

`page 99` **Anomaly Detection.** Most deep learning attempts at anomaly detection involve modeling the normal sample distribution and then identifying any sample that falls outside of this modeled distribution. However, from personal experience, non-deep-learning algorithms tend to work much more effectively and efficiently and require far less tuning, including Isolation Forest by Liu et al. ([2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)), Local Outlier Factor by Breunig et al. ([2000](http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf)) and One-class SVM by Schölkopf et al. ([1999](http://users.cecs.anu.edu.au/~williams/papers/P132.pdf)).

`page 99` **Synthesis and Sampling.** Recent prominent work here for generating images include the VAE ([Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114)), GAN ([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)) and PixelCNN ([van den Oord et al., 2016](https://arxiv.org/abs/1606.05328)) / PixelCNN++ ([Salimans et al., 2017](https://arxiv.org/abs/1701.05517)).

`page 101`

> The most common approach is to report the average log-probability the model assigns to some examples.

In image generation eg. ([van den Oord et al., 2016](https://arxiv.org/abs/1606.05328)), a common metric is the negative log likelihood assigned by the model to a test set of unseen samples. The negative log likelihood is typically normalized by the number of dimensions of the sample, with units bits/dim or bits/subpixel (since each pixel is usually 3 dimensions, 1 per channel for RGB images).

`page 101` Here we touch briefly on the problem of deciding a performance measure. This problem is actually extremely pervasive and relates to AI existential risk. A common example given in reinforcement learning research is when the AI agent learns to exploit a loophole in order to fulfill a poorly-defined objective. For instance, an agent playing Breakout, if trained with the objective of not letting the ball go past the paddle, might learn to pause the game. 

Goodhart's law is appropriate here, where economist Charles Goodhart once stated, "Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes." Or as rephrased by Marilyn Strathern, "When a measure becomes a target, it ceases to be a good measure."

`page 102` Equation 5.1 is often used by autoregressive generative models such as the PixelCNN ([van den Oord et al., 2016](https://arxiv.org/abs/1606.05328)).

`page 104` ToDo: Implement Linear Regression in Python

`page 107` Interesting to learn that the bias parameter `b` is so named because in the absence of any input, the output of the function is biased towards `b`.

`page 108` **i.i.d.** Independent and identically distributed. In this context, it means that each data sample is drawn from the same (identical) distribution and the generation/observation of every sample is independent of every other sample.

`page 109`

> Models with high capacity can solve complex tasks, but when their capacity is higher than needed to solve the *present* task, they may overfit. (emphasis mine)

The keyword here being **present**. In the case of machine learning, the problem of overfitting occurs when the present task that the model is trained on (ie. the training set) is too small and not representative of the data distribution. The model then overfits on the present task in the form of the training set. 

`page 111` **Vapnik-Chervonenkis dimension.** This is a cool concept worth thinking about, but I found the explanation in the book rather lacking. Here's a [link](https://www.quora.com/Explain-VC-dimension-and-shattering-in-lucid-Way) (see the first answer) that I found explained the VC dimension and the concept of shattered sets quite well.

`page 121` It is not mentioned explicitly but I believe the ![m](http://latex.codecogs.com/gif.latex?m) in Equation 5.20 refers to the number of samples drawn from the data distribution in order to calculate or arrive at the estimator ![\hat{\mathbf{\theta}}_m](http://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmathbf%7B%5Ctheta%7D%7D_m).

`page 126` I had a bit of trouble understanding Equations 5.53 and 5.54. To put things into context, we can consider again the linear regression problem from Figure 5.2. Then we can interpret Equation 5.53 as the MSE between the predictions and the true labels for an unseen test set, where the mean/expectation is taken over different choices of the training set. 

For example, with reference to Figure 5.2, we could have ten different training sets and if we fit a degree-1 polynomial to each of these training sets and then calculate the MSE with a single unseen test set, the MSE will be consistently high because the degree-1 polynomial underfits. Since the MSE has a consistent magnitude, there is high bias.

Alternatively, if we fit a degree-9 polynomial to each of the ten training sets and again calculate the MSE with the test set, we will see that the error varies widely depending on the training set, which shows high variance.

The figure from [this](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) Wikipedia article also gives a good illustration.

`page 128`

> For example, it is prone to numerical underflow.

Callback to Chapter 4. Here the underflow occurs if we evaluate across too many data samples, since probability/likelihood ![p](http://latex.codecogs.com/gif.latex?p) is always less than or equal to 1 (although more than or equal to 0) and ![\text{lim}_{m\rightarrow\infty}p^m=0](http://latex.codecogs.com/gif.latex?%5Ctext%7Blim%7D_%7Bm%5Crightarrow%5Cinfty%7Dp%5Em%3D0).

`page 129`

> For example, mean squared error is the cross-entropy between the empirical distribution and a Gaussian model.

This might not be immediately intuitive, but it is explained in the proceeding section, particularly in the Linear Regression as Maximum Likelihood example on page 130. 

Just a summary here:

Suppose we have a function ![p](http://latex.codecogs.com/gif.latex?p) such that for every sample ![\mathbf{x}^{(i)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D), ![p](http://latex.codecogs.com/gif.latex?p) returns the probability of ![y^{(i)}](http://latex.codecogs.com/gif.latex?y%5E%7B%28i%29%7D). We can define ![p](http://latex.codecogs.com/gif.latex?p) as a Gaussian with a constant predefined variance and a sample-dependent mean that is given by another function ![\hat{y}^{(i)}=\hat{y}(\mathbf{x}^{(i)};\mathbf{w})](http://latex.codecogs.com/gif.latex?%5Chat%7By%7D%5E%7B%28i%29%7D%3D%5Chat%7By%7D%28%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D%3B%5Cmathbf%7Bw%7D%29) (such as Equation 5.3). Then, in order to maximize the likelihood of the dataset, it is intuitive that ![\hat{y}^{(i)}](http://latex.codecogs.com/gif.latex?%5Chat%7By%7D%5E%7B%28i%29%7D) must be close to ![y^{(i)}](http://latex.codecogs.com/gif.latex?y%5E%7B%28i%29%7D) (see Equation 5.65).

`page 131`

>That parametric mean squared error decreases as m increases [...]

To clarify, the *parametric mean squared error* refers to the MSE between the learned parameters and the *true* parameters. So as ![m](http://latex.codecogs.com/gif.latex?m), the number of samples, increases, the learned parameters get closer to the *true* parameters.

`page 132` It is great that the Bayesian perspective is covered here, because from personal experience, the Bayesian perspective is less commonly seen in popular research papers. 

Suppose we have an untrained model, the possible values of the model's parameters might be uniformly distributed in a finite parameter space. As we see more training data, we can anneal or concentrate the parameters - certain regions in the parameter space become more likely and certain regions become less likely, by considering the compatibility with the observed training data. When we want to make a prediction, we can then consider the accumulated predictions made by all possible models in this parameter space and weigh the predictions by the learned/annealed likelihood of the parameters.

`page 138` To clarify Equation 5.82, ![\mathbf{x}^{(i)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D) refers to a training sample, while ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D) refers to a test sample or the sample to be predicted. It might be clearer if we realize that, after training on all the training samples, ![\mathbf{w}=\sum^m_{i=1}\alpha_i\mathbf{x}^{(i)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D%3D%5Csum%5Em_%7Bi%3D1%7D%5Calpha_i%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D).

`page 141` On a side note, [here's](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/) a nice visual explanation of decision trees.

`page 149` An interesting comparison of deep learning with SGD to kernel methods. The cost per update is independent of the size of the training data (but dependent on the batch size and size of the model). In contrast, kernel methods typically require computing an m x m matrix where m is the size of the training data (eg. SVM). From that perspective, deep learning is a far more scalable way of training on huge datasets.

 