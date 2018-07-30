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