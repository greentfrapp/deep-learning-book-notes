# 9 Convolutional Networks

`page 323`

> Usually the latter formula is more straightforward to implement in a machine learning library, because there is less variation in the range of valid values of m and n.

See that in Equation 9.5, m and n traverses the size of the kernel, whereas in Equation 9.4, m and n traverses the size of the input. Typically in convlutional neural networks, the kernel is limited to a smaller size eg. 3x3, as compared to the input, which can be upwards of 100x100.

`page 323`

> The commutative property of convolution arises because we have flipped the kernel relative to the input [...]

Specifically, considering Equations 9.4 and 9.5, when the input to I increases, the input to K decreases ie. *flips*. But this is not necessary and is merely done to achieve the commutative property for convenience in proofs and theories. The non-flipping and non-commutative alternative is known as cross-correlation (see Equation 9.6).

`page 324` [Here](https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication) is a nice explanation of how we can view convolution as a regular matrix multiplication (see first answer). Notice that the resulting sparse matrix containing k elements is circulant along rows ie. each row is the preceding row shifted by one to the right. A moment's thought will show that the amount that each row shifts by is dictated by the *stride* of the convolutional layer, in the context of convolutional networks.

`page 325` Figure 9.1 shows 2-D convolution without kernel flipping. For illustration, if kernel-flipping is used, then the first output box (top left) should give [az+by+wf+xe](http://latex.codecogs.com/gif.latex?az&plus;by&plus;wf&plus;xe). See that moving right in the input corresponds to moving left in the kernel and likewise for up/down.

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

A good example is the [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) dataset, which is a visual-question-answering (VQA) dataset, where the model has to answer questions about an image. An example question could be: "What size is the cylinder that is left of the brown metal thing that is left of the big sphere?" We obviously need to preserve spatial information and hence excessive pooling would be detrimental in this case.

`page 336`

> Models that do not use convolution would be able to learn even if we permuted all the pixels in the image.

This is an interesting and important observation that highlights a key difference between convolutional and fully connected networks.

Note: I believe even regular network will have very limited success (ie. probably not work) if we perform random permutations on every image. But it should work if we apply the same permutation to every image.
