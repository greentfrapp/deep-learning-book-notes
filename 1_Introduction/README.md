# 1 Introduction

`page 2` Cyc's story (Linde, 1992) of not recognizing "FredWhileShaving" as a person is used to illustrate a pitfall of the knowledge-base approach to AI. The nature of the mistake is not explicitly described, probably because it seems too ludicrous a mistake and it fits the caricature of dumb inflexible robots in movies. But is it truly a pitfall? Consider prevalent debates on the humanity of cyborgs. Could it be that Cyc's concern here is not mistaken but merely premature? Furthermore, are today's algorithms any better? With reference to Gary Marcus's ([Marcus, 2018](https://arxiv.org/abs/1801.00631)) and numerous papers on adversarial attacks, today's deep learning feels equally brittle, while being far less understandable.

`page 2` 
> AI systems need the ability to acquire their own knowledge

1. **This does not necessarily imply that such systems should learn everything from scratch.** Even current algorithms have inherent biases and priors imposed by design. Convolutional and pooling layers take advantage of positional invariance in image recognition (ie. the position of a dog in an image is far less important than its mere presence in the image). Recurrent architectures take advantage of the sequential dependences. Likewise, to a debatable extent, humans are predisposed to learning and acquiring knowledge in certain manners - we do not learn to strengthen and weaken neurons nor do we learn to touch, hear, smell, see or taste. In support of that, recent successes in reinforcement learning follow the theme of imitation learning ([Salimans & Chen, 2018](https://blog.openai.com/learning-montezumas-revenge-from-a-single-demonstration/)).
2. **Deep Neural Networks are not the only means for AI to acquire knowledge.** In fact, the main knowledge-acquisition mechanism in deep nets is the backpropagation algorithm along with gradient descent. Knowledge acquisition takes the form of weight updates, where knowledge is encoded in the weights matrices of the layers. Other paradigms are possible, such as genetic algorithms, symbolic programming or a combination of symbolic programming and neural networks ([Evans & Grefenstette, 2018](https://deepmind.com/blog/learning-explanatory-rules-noisy-data/)).

`page 4` Representation learning is a solution to the problem of expensive feature-engineering. However, that brings about the problem of interpreting and debugging the learned representations. These are active fields: ([Milli et al., 2018](https://blog.openai.com/interpretable-machine-learning-through-teaching/)) ([Mordvintsev et al., 2018](https://distill.pub/2018/differentiable-parameterizations/)) ([Olah et al., 2018](https://distill.pub/2018/building-blocks/)) ([Olah et al., 2018](https://distill.pub/2017/feature-visualization/))

`page 13` 
> intended to be computational models of biological learning

See Bengio's paper on biologically plausible mechanisms and the NIPS paper on Direct Feedback ([Bengio, 2015](https://arxiv.org/abs/1502.04156)) ([Lillicrap, 2016](https://www.nature.com/articles/ncomms13276)) ([Nøkland, 2016](https://arxiv.org/abs/1609.01596)). Hardware-wise, there is also a huge gap between biological brains and current AI, in terms of power consumption, heat dissipation and size.

`page 14` There are several interesting weight update algorithms eg. WINNOW ([Littlestone, 1988](https://link.springer.com/article/10.1023%2FA%3A1022869011914))

`page 15` As an example on current research in the intersections of machine learning and neuroscience, see Januszewski et al. ([2018](https://ai.googleblog.com/2018/07/improving-connectomics-by-order-of.html))

`page 15` ReLU - note to self: understand more about differences between sigmoid, tanh, relu, leaky ReLU, ELU etc. in terms of considerations such as local linearity ([Goodfellow, 2014](https://arxiv.org/abs/1412.6572)) - later in Chapter 6? Also see work by Ramachandran et al. ([2018](https://arxiv.org/abs/1710.05941)) on searching for activation functions.

`page 16` **Distributed Representation** - Interesting paradigm, never really thought about it this way. It seems extremely nonintuitive to encode the first example (9 combinations) although that might be a bias due to my exposure to present-day deep learning. How does this relate to current deep learning systems?

`page 25` Read Neural Turing Machine paper ([Graves, 2014](https://arxiv.org/abs/1410.5401))
