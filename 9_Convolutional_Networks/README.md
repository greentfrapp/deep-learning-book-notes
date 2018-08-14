# 9 Convolutional Networks

`page 323`

> Usually the latter formula is more straightforward to implement in a machine learning library, because there is less variation in the range of valid values of m and n.

See that in Equation 9.5, m and n traverses the size of the kernel, whereas in Equation 9.4, m and n traverses the size of the input. Typically in convolutional neural networks, the kernel is limited to a smaller size eg. 3x3, as compared to the input, which can be upwards of 100x100.

`page 323`

> The commutative property of convolution arises because we have flipped the kernel relative to the input [...]

Specifically, considering Equations 9.4 and 9.5, when the input to I increases, the input to K decreases ie. *flips*. But this is not necessary and is merely done to achieve the commutative property for convenience in proofs and theories. The non-flipping and non-commutative alternative is known as cross-correlation (see Equation 9.6).

`page 324` [Here](https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication) is a nice explanation of how we can view convolution as a regular matrix multiplication (see first answer). Notice that the resulting sparse matrix containing k elements is circulant along rows ie. each row is the preceding row shifted by one to the right. A moment's thought will show that the amount that each row shifts by is dictated by the *stride* of the convolutional layer, in the context of convolutional networks.

`page 325` Figure 9.1 shows 2-D convolution without kernel flipping. For illustration, if kernel-flipping is used, then the first output box (top left) should give ![az+by+wf+xe](http://latex.codecogs.com/gif.latex?az&plus;by&plus;wf&plus;xe). See that moving right in the input corresponds to moving left in the kernel and likewise for up/down.

`page 327`

> This effect increases if the network includes architectural features like strided convolution (figure 9.12) or pooling (section 9.3).

Another interesting architectural feature is [atrous convolution](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d#b250), which uses a sparse matrix for the kernel ([Chen et al., 2014](https://arxiv.org/abs/1412.7062)).

`page 329` In Figure 9.6, the convolution kernel is simply [-1, 1]. It is a good exercise to try to work out why and also to derive the number of operations. 

While the text mentions that this kernel can be used to detect vertically oriented edges, it can also be helpful to see that the kernel is equivalently a horizontal gradient detector. For any two pixels, if the gradient is positive (increasing) from left to right then the convolution operation will produce a positive value. If the gradient is negative from left to right, then the convolution operation with produce a negative value. If there is no gradient (ie. the two pixels have the same value) the convolution operation will return 0. Also notice that the magnitude of the output indicates the magnitude of the gradient. 

Finally, we also can see that edges can be defined as locally dense gradients.

`page 332`

> (Figure 9.8) Every value in the bottom row has changed, but only half the values in the top row have changed [...]

It might be confusing to some since it seems like both the top and bottom row shifted right by one pixel in the bottom figure, so where's the positional invariance?

Actually, we should be noticing that the second and third outputs still remain as 1. even after the input has shifted, because the maximum input over the neighborhood of 3 pixels is still 1.. In particular, suppose we are pooling over a region of 10 pixels. Then a larger group of output will remain unchanged even after we shift the input.

It is also easy to see that pooling is a rather lossy operation.

`page 332`

> [...] it is possible to use fewer pooling units than detector units, by reporting summary statistics for pooling regions spaced k pixels apart rather than 1 pixel apart.

By using k > 1 spacing/stride, the resulting output will have smaller dimensions, which is why pooling operations are commonly used for downsampling. It is also a good exercise to calculate the output shape as a function of the pooling parameters (kernel size, stride and padding).

`page 335` It is good to look at Figure 9.11 in detail to see how different architectural features can be used together to generate a final output for classification.

`page 336`

> If a task relies on preserving precise spatial information, then using pooling on all features can increase the training error.

A good example is the [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) dataset by Johnson et al. ([2016](https://arxiv.org/abs/1612.06890)), which is a visual-question-answering (VQA) dataset, where the model has to answer questions about an image. An example question could be: "What size is the cylinder that is left of the brown metal thing that is left of the big sphere?" We obviously need to preserve spatial information and hence excessive pooling would be detrimental in this case.

`page 336`

> Models that do not use convolution would be able to learn even if we permuted all the pixels in the image.

This is an interesting and important observation that highlights a key difference between convolutional and fully connected networks.

Note: I believe even regular network will have very limited success (ie. probably not work) if we perform random permutations on every image. But it should work if we apply the same fixed permutation to every image.

`page 337` Here's some additional notes on Equation 9.7 for clarity.

First note that both ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D) and ![\mathbf{Z}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BZ%7D) are channel-first. Also, here we begin the index counting from 1 onwards instead of 0, although 0 is more commonly used in many programming languages.

Suppose we want to calculate ![Z_{1,1,1}](http://latex.codecogs.com/gif.latex?Z_%7B1%2C1%2C1%7D):

![Z_{1,1,1}=\sum_{l,m,n}V_{l,1+m-1,1+n-1}K_{1,l,m,n}](http://latex.codecogs.com/gif.latex?Z_%7B1%2C1%2C1%7D%3D%5Csum_%7Bl%2Cm%2Cn%7DV_%7Bl%2C1&plus;m-1%2C1&plus;n-1%7DK_%7B1%2Cl%2Cm%2Cn%7D)

First, see that the kernel weights required for calculating ![Z_{1,1,1}](http://latex.codecogs.com/gif.latex?Z_%7B1%2C1%2C1%7D) are all present in ![K_{1,:,:,:}](http://latex.codecogs.com/gif.latex?K_%7B1%2C%3A%2C%3A%2C%3A%7D), since the first index in ![\mathbf{K}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BK%7D) corresponds to the output channel.

Next, in order to calculate ![Z_{1,1,1}](http://latex.codecogs.com/gif.latex?Z_%7B1%2C1%2C1%7D), what elements from ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D) do we need? If ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D) has only 1 channel and if the kernel is of shape (2, 2), then we just need to consider the top left (2, 2) square in ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D), which comprises 4 values, in order to calculate ![Z_{1,1,1}](http://latex.codecogs.com/gif.latex?Z_%7B1%2C1%2C1%7D). If ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D) has 3 channels, then we need to consider the same top left (2, 2) square in ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D) but now we have to look across the 3 channels, which means we need to consider 12 elements in ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D).

We see that the number of elements to consider from ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D) is determined by the number of channels in ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D), as well as the kernel size of ![\mathbf{K}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BK%7D), given by the range of the last 2 indices in ![\mathbf{K}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BK%7D).

In the above equation, ![l](http://latex.codecogs.com/gif.latex?l) shows the summation over the channels in ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D), while ![m](http://latex.codecogs.com/gif.latex?m) and ![n](http://latex.codecogs.com/gif.latex?n) show the summation over the kernel size of ![\mathbf{K}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BK%7D).

If we work this through all the way, we can see that as we approach the right and bottom part of ![\mathbf{Z}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BZ%7D), we might end up out of range of ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D). Implementation-wise, we perform zero padding in order to account for this. We actually pad all around the input so the ![Z_{1,1,1}](http://latex.codecogs.com/gif.latex?Z_%7B1%2C1%2C1%7D) should have started with a zero padded ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D) as well.

`page 338` Equation 9.8 can be easily derived as an extension of Equation 9.7 and the derivation is also a good exercise to understand how convolution works in convolutional networks.

`page 338` It is a good exercise to try to derive the output dimensions of the different padding methods.

The terms 'valid' and 'same' are also used in Tensorflow (eg. in [`tf.layers.conv2d`](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)).

For clarification, in zero padding for *full* convolution, we add m-1 zeros at the top and bottom and n-1 zeros at the left and right of the input, for a kernel of shape (m, n) (height m and width n). This means that every element in the input (including the border elements) is visited m+n times. In order to generate, say, the top left output, we only require the top left input, with the rest being padded with zeros.

`page 346` For Equation 9.11, we need to see that the right part of the equation is actually a formulation of the chain rule where ![\frac{\partial J(\mathbf{V},\mathbf{K})}{\partial {K_{i,j,k,l}}}=\sum_{m,n}\frac{\partial J(\mathbf{V},\mathbf{K})}{\partial {Z_{i,m,n}}}\frac{\partial {Z_{i,m,n}}}{\partial K_{i,j,k,l}}](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20J%28%5Cmathbf%7BV%7D%2C%5Cmathbf%7BK%7D%29%7D%7B%5Cpartial%20%7BK_%7Bi%2Cj%2Ck%2Cl%7D%7D%7D%3D%5Csum_%7Bm%2Cn%7D%5Cfrac%7B%5Cpartial%20J%28%5Cmathbf%7BV%7D%2C%5Cmathbf%7BK%7D%29%7D%7B%5Cpartial%20%7BZ_%7Bi%2Cm%2Cn%7D%7D%7D%5Cfrac%7B%5Cpartial%20%7BZ_%7Bi%2Cm%2Cn%7D%7D%7D%7B%5Cpartial%20K_%7Bi%2Cj%2Ck%2Cl%7D%7D).

The partial derivative of each parameter (in ![\mathbf{K}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BK%7D)) is then given by the input element (from ![\mathbf{V}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BV%7D)) that it was multipled by.

A similar intuition can be applied for Equation 9.13.

`page 347`

> This allows the model to label every pixel in an image and draw precise masks that follow the outlines of individual objects.

This refers to image segmentation and the technique where each channel/layer in the output tensor refers to a class is used in architectures such as Fully Convolutional Networks by Long et al. ([2015](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)).

`page 351`

> When a d-dimensional kernel can be expressed as the outer product of d vectors, one vector per dimension, the kernel is called separable.

For clarity, the *d-dimensional* here refers to the size of the shape of the kernel. For instance, a kernel of shape (3, 3) is 2-dimensional. It might be possible to express such a kernel as an outer product of 2 vectors, each of size 3. If that is true, it would be more efficient to store it as 2 vectors, which gives a total of 6 elements rather than to store it as 9 elements in the original kernel. 

Furthermore, instead of performing convolution with the original kernel, it will be faster (require less operations) to perform 1-dimensional convolution with each of the 2 vectors (first perform 1-dimensional convolution between the input and 1 of the 2 vectors, then perform 1-dimensional convolution between the resulting output and the second vector).

`page 355`

> In the context of deep learning, attention mechanisms have been most successful for natural language processing [...]

In an interestingly-named paper titled "Attention is All You Need", Vaswani et al. ([2017](https://arxiv.org/abs/1706.03762)) introduced an architecture known as the Transformer, which actually uses only attention and regular feedforward networks to achieve SOTA machine translation performance.

Apart from natural language processing, attention has also been used in tasks involving a combination of language and computer vision, such as image captioning ([Xu et al., 2015](https://arxiv.org/abs/1502.03044)).

`page 361`

> This approach has been the most successful on a two-dimensional image topology.

It is true that convolutional networks have demonstrated the most success in computer vision. But, as mentioned in previous pages, 1-D sequential data (eg. audio waveforms) can also be processed using convolutional networks. 

One interesting concern for sequential data is causal dependency ie. the future cannot influence the past. This means that if we were to apply convolutional models, we have to be careful not to violate this dependency and accidentally allow future time steps to be factored into modeling previous time steps.

A nice example of convolutional networks used on sequential data is WaveNet by van den Oord et al. ([2016](https://arxiv.org/abs/1609.03499)), which explicit considers the causal dependency in the data.
