# 10 Sequence Modeling: Recurrent and Recursive Nets

`page 363`

> Such sharing is particularly important when a specific piece of information can occur at multiple positions within the sequence.

Another interesting example is the following pair of sentences:

> I am not sad, I'm happy.

> I am sad, I'm not happy.

In this case, the position of the word 'not' is critical to the understanding of the sentence. As such, any model used to process such sentences should not be position invariant, unlike what we saw before in CNNs. But this might not be the case for all types of sequential data.

`page 368` **Recurrent Neural Networks.** Andrej Karpathy has a great [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) on character-level RNNs trained on several different datasets.

`page 368` Figure 10.5 (page 370) is actually the same as Figure 10.3, in that we *are* able to get an output from the hidden state at every step. But in the case of Figure 10.5, we disregard all the hidden states except for the last one, which we pass to another network to generate a prediction.

`page 370` The architecture in Figure 10.6 is less powerful than in Figure 10.3, because we assume that the output dimension is significantly smaller than the hidden dimension. As such, using the output from the previous step as a representation of the past is far more lossy than directly using the hidden state. Furthermore, the output is optimized to match the label via the loss function. So unless the label contains the same amount of information as the hidden state, there will definitely be information loss if we only rely on passing the output from one step to the next.

`page 370`

> Forward propagation begins with a specification of the initial state ![\mathbf{h}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%7B%280%29%7D).

A question might be how we should initialize ![\mathbf{h}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%7B%280%29%7D). Implementation-wise, it does not really matter how we initialize ![\mathbf{h}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%7B%280%29%7D) so long as it is fixed ie. the same ![\mathbf{h}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%7B%280%29%7D) is used when consuming every new sequence, so it can just be set to all zeros. 

Alternatively, we can also set it as a trainable parameter to be optimized.

`page 373` **Teacher Forcing.** An example of teacher forcing is when we train a network to perform machine translation, using Sequence-to-Sequence architectures (see page 385). Suppose we are translating French to German. In order to predict the next German word in a sequence, the model has to consider the input French sentence, as well as the German words translated so far. 

During testing, the German words translated so far refer to the previous predictions made by the model. However, during training, we can choose to provide the model with the correctly translated previous German words (ground truth), rather than use its previous predictions.

If we use the model's previous predictions during training, if the model makes a wrong prediction, this error prevents subsequent predictions from being trained correctly, since the model is being conditioned on the wrong words.

By using the ground truth (ie. teacher forcing), the model is able to train correctly (conditioned on the correct German words), even if its initial predictions are wrong.

One problem with this approach (explained in the next page) is that during testing, the model might see a different set of inputs, since it is always conditioned on the correct inputs during training. The model is not trained to 'recover' in the case that it makes a mistake in its initial prediction and has to make subsequent predictions conditioned on its previous mistakes.

Some techniques to mitigate this are discussed in the next page.

`page 374` For the derivation of BPTT, it is good to refer to Figure 10.6 and Equations 10.8 to 10.11 for the notations. In addition ![\mathbf{b}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bb%7D) is the bias used in calculating ![\mathbf{h}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D), while ![\mathbf{c}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bc%7D) is the bias used in calculating ![\mathbf{o}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bo%7D).

For Equation 10.17, note that ![L](http://latex.codecogs.com/gif.latex?L) refers to a summation over ![L^{(t)}](http://latex.codecogs.com/gif.latex?L%5E%7B%28t%29%7D) (see Equation 10.13).

For Equation 10.18, I believe the ![\mathbf{1}_{i,y^{(t)}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B1%7D_%7Bi%2Cy%5E%7B%28t%29%7D%7D) refers to 1 if ![y^{(t)}_i=i](http://latex.codecogs.com/gif.latex?y%5E%7B%28t%29%7D_i%3Di) and 0 otherwise.

`page 375`

> This is the Jacobian of the hyperbolic tangent associated with the hidden unit ![i](http://latex.codecogs.com/gif.latex?i) at time ![t+1](http://latex.codecogs.com/gif.latex?t&plus;1).

Equation 10.9 gives the tanh relation and the gradient for tanh is given as: ![\frac{\mathrm{d}\tanh(x)}{\mathrm{d}x}=1-\tanh^2(x)](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cmathrm%7Bd%7D%5Ctanh%28x%29%7D%7B%5Cmathrm%7Bd%7Dx%7D%3D1-%5Ctanh%5E2%28x%29).

`page 377`

> For example, it is common to make the Markov assumption that the graphical model should contain only edges from ![\{y^{(t-k)},\cdots,y^{(t-1)}\}](http://latex.codecogs.com/gif.latex?%5C%7By%5E%7B%28t-k%29%7D%2C%5Ccdots%2Cy%5E%7B%28t-1%29%7D%5C%7D) to ![y^{(t)}](http://latex.codecogs.com/gif.latex?y%5E%7B%28t%29%7D), rather than containing edges from the entire history.

This refers to kth-order Markov models. For example, a 1st-order Markov model means that ![y^{(t)}](http://latex.codecogs.com/gif.latex?y%5E%7B%28t%29%7D) only depends on ![y^{(t-1)}](http://latex.codecogs.com/gif.latex?y%5E%7B%28t-1%29%7D), while a 2nd-order Markov model means that ![y^{(t)}](http://latex.codecogs.com/gif.latex?y%5E%7B%28t%29%7D) only depends on ![y^{(t-1)}](http://latex.codecogs.com/gif.latex?y%5E%7B%28t-1%29%7D) and ![y^{(t-2)}](http://latex.codecogs.com/gif.latex?y%5E%7B%28t-2%29%7D).

`page 379`

> one can add a special symbol corresponding to the end of the sequence

Note that this *special symbol* should not be in the original vocabulary of the dataset ie. its only role is to represent the end of the sequence.