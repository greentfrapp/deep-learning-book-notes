# 12 Applications

`page 432` [Here](https://stackoverflow.com/questions/7524838/fixed-point-vs-floating-point-number) is a simple explanation of the difference between fixed- and floating-point arithmetic (refer to the first answer).

`page 433` [Here](https://www.quora.com/Why-are-GPUs-more-powerful-than-CPUs) is an excellent metaphor for the difference between GPU and CPU (refer to the first answer).

`page 436` **Compression Networks.** This is a domain that has received increasing attention, largely due to commercial interest. The main aim of compression networks is to minimize the computation and memory resources required for implementing neural networks (training and inference). 

We saw in Chapter 7 that L1-regularization can lead to sparse weight matrices. Storing these weights as sparce matrices can also help reduce the memory requirements of the network. There has also been some work done on sparse networks using L0-regularization ([Louizos et al., 2018](https://arxiv.org/abs/1712.01312)).

Another popular technique is distillation, first introduced by Hinton et al. ([2015](https://arxiv.org/abs/1503.02531)), where a larger network is used to train a smaller network. This was followed up by a work by Frosst & Hinton ([2017](https://arxiv.org/abs/1711.09784)), which applied the same concept but distilling networks to decision trees.

Some libraries also have built-in compression capabilities such as [quantization in TensorFlow Lite](https://www.tensorflow.org/performance/quantization).

`page 437` **Cascading.** This is an interesting idea that can possibly be exploited for anomaly detection.

`page 438` **Mixture of Experts.** Despite its 'age', the mixture of experts algorithm () is still used actively in modern research, such as in the YouTube-8M challenge ([Abu-El-Haija et al., 2016](https://arxiv.org/abs/1609.08675)). [Here](https://www.youtube.com/watch?v=2G99dq7ccqc) is a great YouTube video of Hinton explaining the Mixture of Experts algorithm.

`page 440`

> [...] recognize sound waves from the vibrations they induce in objects visible in a video [...]

This is actually a pretty fascinating work, which also has some severe implications for privacy. [Here](https://dl.acm.org/citation.cfm?id=2601119) is the link to the paper by Davis et al. There is also an accompanying YouTube [video](https://www.youtube.com/watch?v=FKXOucXB4a8).

`page 441`

> While image synthesis *ex nihilo* is usually not considered a computer vision endeavor, models capable of image synthesis are usually useful for image restoration [...]

*Ex nihilo* is Latin for *from nothing*. 

While there are many works on image synthesis, restoration and completion, [this](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/) work in particular by Iizuka et al. ([2017](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf)) blew my mind when I first saw it.

On a similar note, Chaitanya et al. ([2017](https://research.nvidia.com/publication/interactive-reconstruction-monte-carlo-image-sequences-using-recurrent-denoising)) also published a great paper at SIGGRAPH 2017 (same venue as the above by Iizuka et al.) on recurrent denoising autoencoders. A video demo can be seen [here](https://www.youtube.com/watch?v=9yy18s-FHWw). On a side note, this technology seems related to demos for the new Nvidia RTX GPUs.

`page 441`

> Many computer vision architectures require images of a certain size, so images must be cropped or scaled to fit that size.

Interestingly, a recent achievement by [fast.ai](https://www.fast.ai) on training 'Imagenet to 93% accuracy in just 18 minutes' made use of rectangular images for validation (rather than traditional square center crops), which gave 'an immediate speedup of 23% in the amount of time it took to reach the benchmark accuracy of 93%'. See the full blog post [here](http://www.fast.ai/2018/08/10/fastai-diu-imagenet/).

`page 445` There is an interesting paper by Cubuk et al. ([2018](https://arxiv.org/abs/1805.09501)) that used reinforcement learning to find optimal image transformation policies, known as AutoAugment. [Here](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html) is the blog post.
