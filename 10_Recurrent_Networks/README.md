# 10 Sequence Modeling: Recurrent and Recursive Nets

`page 363`

> Such sharing is particularly important when a specific piece of information can occur at multiple positions within the sequence.

Another interesting example is the following pair of sentences:

> I am not sad, I'm happy.

> I am sad, I'm not happy.

In this case, the position of the word 'not' is critical to the understanding of the sentence. As such, any model used to process such sentences should not be position invariant, unlike what we saw before in CNNs. But this might not be the case for all types of sequential data.

`page 368` **Recurrent Neural Networks.** Andrej Karpathy has a great [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) on character-level RNNs trained on several different datasets.

`page 368` Figure 10.5 (page 370) is actually the same as Figure 10.3, in that we *are* able to get an output from the hidden state at every step. But in the case of Figure 10.5, we disregard all the hidden states except for the last one, which we pass to another network to generate a prediction.

`page 370` The architecture in Figure 10.6 is less powerful than in Figure 10.3, mainly because we assume that the output dimension is significantly smaller than the hidden dimension. As such, using the output from the previous step as a representation of the past is far more lossy than directly using the hidden state.

`page 370`

> FOrward propagation begins with a specification of the initial state ![\mathbf{h}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%7B%280%29%7D).

A question might be how we should initialize ![\mathbf{h}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%7B%280%29%7D). Implementation-wise, it does not really matter how we initialize ![\mathbf{h}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%7B%280%29%7D) so long as it is fixed ie. the same ![\mathbf{h}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%7B%280%29%7D) is used when consuming every new sequence, so it can just be set to all zeros. 

Alternatively, we can also set it as a trainable parameter to be optimized.

