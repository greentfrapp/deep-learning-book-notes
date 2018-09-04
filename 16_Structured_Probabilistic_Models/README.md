# 16 Structured Probabilistic Models for Deep Learning

`page 549` I first studied probabilistic models from Pearl & Mackenzie's *The Book of Why* ([2018](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X)), which gave an overview of causality principles, facilitated by directed graphical models. It is a great book that provides historical, intuitive and theoretical analyses and explanations for causality and its affiliated concepts, including causal inference, correlation, confounding etc. 

`page 550` For clarification, I believe Density Estimation here refers to (in layman terms) the probability of sample X appearing under the data-generating distribution. For example, if the data-generating distribution is a regular die roll, then the probability of each of the faces appearing is 1/6. This is considerably more complex for distributions such as text and images, but we can still have an intuitive sense of relative probability. For instance, under a regular English model, the sentence 'The dog is fat.' is definitely more probable than the sentence 'The dog corn fat.', since the latter does not make much sense compared to the former.

`page 551` The work on Image Transformers by Parmer et al. ([2018](https://arxiv.org/abs/1802.05751)) gives a good example of missing value imputation, where a trained model is used to complete photos with their bottom half removed.

`page 553` The relay race example here seems similar to a Markov model, where there is a sequence of events (the order of the relay here) and each event (finishing time of each runner) can be modeled as being dependent on only the previous event (finishing time of previous runner).

`page 554` The term 'acyclic' in 'directed acyclic graphs' refers to there being no loop within the graph ie. if you choose any node and follow the direction of the arrows in the graph, you should never return to a node you have reached before.

`page 554` The footnote suggestion from Judea Pearl recalls the discussion in Chapter 3 of the distinction between frequentist and Bayesian probability (see page 53 in hard copy).

`page 559` One way to think about the problem of diverging domain in calculating the partition function - Z is simply the area under ![\phi(x)](https://latex.codecogs.com/gif.latex?%5Cphi%28x%29), so that normalizing by Z will make the area under ![p(x)](https://latex.codecogs.com/gif.latex?p%28x%29), ![\int p(x)dx=1](https://latex.codecogs.com/gif.latex?%5Cint%20p%28x%29dx%3D1). With a diverging domain and an unbounded and continuous ![x](https://latex.codecogs.com/gif.latex?x), area under the curve ![Z=\infty](https://latex.codecogs.com/gif.latex?Z%3D%5Cinfty) and normalizing by that will give ![p(x)=0](https://latex.codecogs.com/gif.latex?p%28x%29%3D0) everywhere, which is obviously incorrect.

`page 565` The various examples in Figures 16.8 and 16.9 are actually described in fascinating detail in Pearl & Mackenzie's *The Book of Why* ([2018](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X)), mentioned above!

`page 567` In simpler terms, in moralization, for every node pair X and Y, we connect them if they are already connected in the directed graph or if they are both parents of a collider (see Figure 16.8 or *The Book of Why* ([Pearl & Mackenzie, 2018](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X)) for explanation of a collider).

`page 570` Ancestral sampling is closely related to autoregressive modeling, where an event is modeled as being dependent on previous events. For example, in language modeling, each word in a sentence is dependent on previous words, while in image modeling, each pixel may be modeled as being dependent on previous pixels (the order does not really matter, so long as it is fixed and consistent for training and testing). See this [blogpost](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy for character-level language modeling. The PixelCNN by van den Oord et al. ([2016](https://arxiv.org/pdf/1606.05328.pdf)) is also a popular autoregressive model for image modeling.

`page 573`

> The new variables ![\mathbf{h}](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D) also provide an alternative representation for ![\mathbf{v}](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bv%7D).

This would be closely related to the discussions on Representation Learning in Chapter 15.

`page 574` Equation 16.9 is just a reformulation of Bayes' Rule, where

![p(\mathbf{v}\mid\mathbf{h})=\frac{p(\mathbf{h}\mid\mathbf{v})p(\mathbf{v})}{p(\mathbf{h})}](https://latex.codecogs.com/gif.latex?p%28%5Cmathbf%7Bv%7D%5Cmid%5Cmathbf%7Bh%7D%29%3D%5Cfrac%7Bp%28%5Cmathbf%7Bh%7D%5Cmid%5Cmathbf%7Bv%7D%29p%28%5Cmathbf%7Bv%7D%29%7D%7Bp%28%5Cmathbf%7Bh%7D%29%7D)

Then, we rearrange the terms and substitute ![p(\mathbf{h},\mathbf{v})=p(\mathbf{h}\mid\mathbf{v})p(\mathbf{v})](https://latex.codecogs.com/gif.latex?p%28%5Cmathbf%7Bh%7D%2C%5Cmathbf%7Bv%7D%29%3Dp%28%5Cmathbf%7Bh%7D%5Cmid%5Cmathbf%7Bv%7D%29p%28%5Cmathbf%7Bv%7D%29), to get

![p(\mathbf{v})=\frac{p(\mathbf{h},\mathbf{v})}{p(\mathbf{h}\mid\mathbf{v})}](https://latex.codecogs.com/gif.latex?p%28%5Cmathbf%7Bv%7D%29%3D%5Cfrac%7Bp%28%5Cmathbf%7Bh%7D%2C%5Cmathbf%7Bv%7D%29%7D%7Bp%28%5Cmathbf%7Bh%7D%5Cmid%5Cmathbf%7Bv%7D%29%7D)

Then we just apply the log on both sides to get Equation 16.9.

`page 574` SAT problems are also known as Boolean satisfiability problems, where the aim is to determine if a given Boolean formula is satisfiable ie. whether the formula can be evaluated as True given a certain assignment of the variables. To quote an example from the Wikipedia [article](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem):

> For example, the formula "a AND NOT b" is satisfiable because one can find the values a = TRUE and b = FALSE, which make (a AND NOT b) = TRUE. In contrast, "a AND NOT a" is unsatisfiable. 

The 3-SAT problem then refers to SAT problems where each clause in the Boolean formula has at most 3 literals (see the Wikipedia [page](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem) for more information). On a side note, the 3-SAT problem is also one of Richard Karp's [21 NP-complete problems](https://en.wikipedia.org/wiki/Karp%27s_21_NP-complete_problems).

`page 575`

> The latent variables are usually not very easy for a human to interpret after the fact, though visualization techniques may allow some rough characterization of what they represent.

The Distill journal has several really great articles on feature visualization ([Olah et al., 2017](https://distill.pub/2017/feature-visualization/)) ([Olah et al., 2018](https://distill.pub/2018/building-blocks/)) ([Mordvintsev et al., 2018](https://distill.pub/2018/differentiable-parameterizations/)).

`page 577` **Restricted Boltzmann Machine.** [Here](http://deeplearning.net/tutorial/rbm.html) is the link to the LISA Lab tutorial on RBMs, complete with Python code for training with the Theano library.

`Exercise` As a simple exercise, it will be good to check out Figure 1 in Finn et al.'s work ([2018](https://arxiv.org/abs/1806.02817)), which discusses metalearning in the context of probabilistic models. The figure also gives an interesting perspective on how learned model parameters can be modeled as dependent on the training data.
