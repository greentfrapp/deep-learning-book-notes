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

`page 445` Regarding data augmentation, there is an interesting paper by Cubuk et al. ([2018](https://arxiv.org/abs/1805.09501)) that used reinforcement learning to find optimal image transformation policies, known as AutoAugment. [Here](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html) is the blog post.

`page 448` Another contemporary work on "aligning" acoustic-level information with phonetic-level information using attention is the work by Chan et al. ([2015](https://arxiv.org/abs/1508.01211)). This is also described briefly by Olah & Carter ([2016](https://distill.pub/2016/augmented-rnns/)) in their Distill article on augmented RNNs.

`page 449` **n-grams.** Previous NLP classification algorithms use bag-of-words (BoW) models where a sentence might be expressed as a vector indicating the appearance of each word.

Suppose we have a vocabulary containing the words ['cats', 'dogs', 'are', 'fat']. Then the sentence 'dogs are fat' might be represented by the vector [0, 1, 1, 1], while the sentence 'cats are fat' might be represented by the vector [1, 0, 1, 1]. 

There are also bag-of-n-gram (BoNG) models where the vocab includes n-grams. For example, if we include up to bigrams (n = 2), the same example from above might now have the following vocabulary ['cats', 'dogs', 'are', 'fat', 'cats dogs', 'dogs are', 'are fat', 'dogs cats', ...] of size 20. The sentence 'dogs are fat' might then be represented by [0, 1, 1, 1, 0, 1, 1, 0,...], since both bigrams 'dogs are' and 'are fat' appear in the sentence. 

From this example, we can also see how n-gram models are susceptible to the curse of dimensionality and how most n-grams will not occur in the training set.

`page 449` **Language Modeling.** I mentioned this in previous notes on Chapter 10 but [here's](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) a great post by Andrej Karpathy on training character-level language models. 

More interestingly, recent research has shown that language models trained on huge amounts of data can actually be used for a wide variety of tasks after some finetuning, including commonsense reasoning! See works by Trinh & Le ([2018](https://arxiv.org/abs/1806.02847)) and Radford et al. ([2018](https://blog.openai.com/language-unsupervised/)).

`page 452` **Word Embeddings.** Popular word-embedding models include Word2Vec ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)) and GloVe ([Pennington et al., 2014](https://nlp.stanford.edu/pubs/glove.pdf)). There is a nice demo [here](http://bionlp-www.utu.fi/wv_demo/) and a more visual one [here](https://lamyiowce.github.io/word2viz/) (can take several minutes to load).

On a side note, word embeddings have also been the focus of algorithmic bias studies such as this work by Bolukbasi et al. ([2016](https://arxiv.org/abs/1607.06520)) and the related MIT Tech Review [article](https://www.technologyreview.com/s/602025/how-vector-space-mathematics-reveals-the-hidden-sexism-in-language/). More precisely, the word embeddings can surface bias found in the training data itself, such as co-occurences that might indicate discriminatory sentiments.

`page 460` **Machine Translation.** The current SOTA in machine translation uses the attention model (described in the next section), known as the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). An implementation can be found [here](https://github.com/tensorflow/tensor2tensor). Vaswani et al.'s work builds on the attention mechanism introduced by Bahdanau et al. ([2014](https://arxiv.org/abs/1409.0473)), which is mentioned in the text. Interestingly, compared to Bahdanau et al., the Transformer architecture by Vaswani et al. does not use any recurrent model, only fully connected networks and multihead dot-product attention.

I have previously worked on a guide to attention and the Transformer model, which can be found [here](https://github.com/greentfrapp/attention-primer).

`page 464` Note: SVD here refers to Singular-Value Decomposition. 

`page 465` On a related note to recommender systems and language modeling, there was an interesting work by Radford et al., ([2017](https://blog.openai.com/unsupervised-sentiment-neuron/)), which found that a character-level language model trained on 82 million Amazon reviews contained a sentiment neuron - a neuron that predicted the sentiment of a review ie. positive or negative. The [blogpost](https://blog.openai.com/unsupervised-sentiment-neuron/) has great examples of some cool experiments done with the sentiment neuron.

`page 468` **Exploration versus Exploitation.** As mentioned in the text, this paradigm comes up frequently in reinforcement learning. Here is a nice way to think about it. Suppose you are in a foreign city and you know that you like the food at Restaurant A (maybe you have been to this city once at tried the food there before). The question is - should you **explore** unknown restaurants or **exploit** your knowledge and go back to Restaurant A? In the former case, you risk eating at worse restaurants but you might also chance upon a restaurant that's much better than Restaurant A. In the latter case, settle for Restaurant A, never knowing whether there might be something better out there. This is also related to the [Secretary problem](https://en.wikipedia.org/wiki/Secretary_problem).

`page 469`

> One of the most prominent factors is the time scale we are interested in.

This also applies intuitively to our restaurant example above. If we know we are going to stay in the foreign city for a year, we are probably better off exploring new restaurants, since the positive effect of the new knowledge can be exploited over a longer term. In contrast, if we are going to be in this foreign city for a 2-hour layover, it might be better to just stick to Restaurant A as a safe bet.

`page 470` The relation discussion here is somewhat related to inductive logic programming (ILP) ([Muggleton & De Raedt, 1994](https://www.sciencedirect.com/science/article/pii/0743106694900353)), which focused on teaching machines to reason with predicates and clauses. More recently, a team from DeepMind introduced ∂ILP, which augmented ILP with deep learning ([Evans & Grefenstette, 2018](https://arxiv.org/abs/1711.04574)). This is actually a pretty good read as it touches on some of the disadvantages of neural networks, such as brittleness and the lack of interpretability.
