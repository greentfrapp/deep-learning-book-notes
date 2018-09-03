# 15 Representation Learning

`page 521`

> First, it makes use of the idea that the choice of initial parameters for a deep neural network can have a significant regularizing effect on the model [...]

This is because we use the pretrained network weights as the initialization of the neural network for subsequent training.

On a side note, this is somewhat relevant to the discussion of *catastrophic forgetting*, where a neural network that is first trained on dataset A and then trained with dataset B will 'forget' about dataset A. The idea of representation learning as initialization seems to run somewhat counter to catastrophic forgetting.

`page 523`

> Words represented by one-hot vectors are not very informative because every two distinct one-hot vectors are the same distance from each other [...]

In fact, every pair of one-hot vectors are the same distance from each other in both Euclidean and cosine distance, since they are orthonormal.

On a related note, in most NLP works, pretrained word embeddings (typically Word2Vec or GloVe) are used instead of one-hot vectors. These word embeddings are a form of unsupervised representation learning, embedding the meaning of the words into their vector representations.

`page 527` This was mentioned in Chapter 12's notes, but on a related note to sentiment analysis and representation learning, Radford et al.'s work ([2017](https://blog.openai.com/unsupervised-sentiment-neuron/)) has some interesting findings.

`page 528` **Concept Drift.** This is not mentioned explicitly in the text but concept drift refers to the phenomena where data or target distributions shift over time. 

A simple example is the use of machine learning to predict shop sales, where a trained model might become increasingly incorrect and irrelevant over time if it doesn't get updated, due to fads or seasonal changes. 

Another example is in reinforcement learning. Since the agent's policy affects the environment, the agent learning and updating its policy directly results in a changing environment with shifting data distribution.

`page 529` Few-shot, one-shot or zero-shot learning are encompassed in a recently emerging field known as **meta-learning**. While traditionally including mainly classification, recent works in meta-learning have included regression and reinforcement learning ([Vinyals et al., 2016](https://arxiv.org/abs/1606.04080)) ([Andrychowicz et al., 2016](https://arxiv.org/abs/1606.04474)) ([Ravi & Larochelle, 2017](https://openreview.net/forum?id=rJY0-Kcll)) ([Duan et al., 2017](https://arxiv.org/abs/1611.02779)) ([Finn et al., 2017](https://arxiv.org/pdf/1703.03400.pdf)). Works in this area seems to be primarily motivated by the notion of human-level AI, since humans appear to be able to require far fewer training data than most deep learning models.

`page 534` **Generative Adversarial Network (GAN).** This has been one of the most popular models (research-wise) in recent years ([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)). This OpenAI [blogpost](https://blog.openai.com/generative-models/) gives a good overview of a few architectures based on the GAN (although only from OpenAI). Other interesting models include pix2pix ([Isola et al., 2017](https://arxiv.org/abs/1611.07004)), CycleGAN ([Zhu et al., 2017](https://arxiv.org/abs/1703.10593)) and WGAN ([Arjovsky et al., 2017](https://arxiv.org/abs/1701.07875)). The first two deal with image-to-image translation (eg. photograph to Monet/Van Gogh or summer photo to winter photo), while the last work focuses on using Wasserstein distance as a metric for stabilizing the GAN (since GANs are known to be unstable and difficult to train).

`page 536` There is an errata in my hardcopy for Figure 15.6, which has been pointed out in the [errata list](https://docs.google.com/document/d/1ABlp7FluwZ0B82_fjNOFVQ2uOZkfuF8elbofhZmNXag/edit). As I quote, 'If the original images are A, B, C, then this was printed as B, C, A.' This has also been rectified in the web version of [Chapter 15](http://www.deeplearningbook.org/contents/representation.html) (see page 543).

`page 538` VC dimension here refers to Vapnik-Chervonenkis dimension, which was mentioned in Chapter 5 (page 111 for hardcopy).

`page 542` As a counter-example, recent research from DeepMind ([Morcos et al., 2018](https://arxiv.org/abs/1803.06959)) suggests that while some hidden units might appear to learn an interpretable feature, 'these interpretable neurons are no more important than confusing neurons with difficult-to-interpret activity'. Moreover, 'networks which generalise well are much less reliant on single directions [ie. hidden units] than those which memorise'. See more in the DeepMind [blog post](https://deepmind.com/blog/understanding-deep-learning-through-neuron-deletion/).

`page 545` Under the Linearity point, the cited work by Goodfellow (mentioned a few times previously) discusses the problem of adversarial examples for neural networks and proposes the hypothesis that the problem is due to the excessive linearity in the models ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)). It is a great read and a good discussion on adversarial examples, as well as the nature of deep neural networks in general.
