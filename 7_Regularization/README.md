# 7 Regularization for Deep Learning

`page 222` Recall that Section 5.4 discussed the bias-variance trade-off in the context of underfitting and overfitting. A low-capacity estimator has higher bias due to underfitting, while a high-capacity estimator has higher variance due to overfitting. In regularization, we try to minimize the variance due to overfitting, while avoiding too large an increase in bias.

`page 222`

> Instead, we might find - and indeed in practical deep learning scenarios, we almost always do find - that the best fitting model (in the sense of minimizing generalization error) is a large model that has been regularized appropriately.

One interesting way to think about it is to consider how human inference functions in a similar manner. Suppose a human is given the following sequence `1, 2, 3`. There is a good chance that the human will predict `4` for the subsequent value, predicting the function as ![y_i=i](http://latex.codecogs.com/gif.latex?y_i%3Di).

But it is also possible to infer more complex functions that result in very different predictions. For instance, the sequence might be following a function where the output at each step ![y_i=\frac{6}{11}i-\frac{1}{11}i^2+\frac{6}{11}](http://latex.codecogs.com/gif.latex?y_i%3D%5Cfrac%7B6%7D%7B11%7Di-%5Cfrac%7B1%7D%7B11%7Di%5E2&plus;%5Cfrac%7B6%7D%7B11%7D). In that case, the next value should be predicted as ![\frac{14}{11}](http://latex.codecogs.com/gif.latex?%5Cfrac%7B14%7D%7B11%7D) or 1.27. 

However, humans tend to prefer the first conclusion for many reasons, amongst which because it is easier to infer and remember. 

Cognitively, humans are able to arrive at both conclusions or even infinite other functions. However, there seems to be a bias towards simpler conclusions (where the notion of *simplicity* needs to be better defined). In that sense, human minds can also be considered large models (capable of considering complex concepts and inferences) that are regularized appropriately (via their bias for simpler concepts and inferences).

`page 223` Note on typically only regularizing weights and not biases.

`page 224` Interesting footnote on regularizing the parameters to be near any arbitrary point rather than just the origin, although there doesn't seem to be a citation for this.

On that note, we can generalize the regularization term to regularize and bias the parameters towards any arbitrary point with the following:

![\Omega(\theta)=\frac{1}{2}\left\|\mathbf{w-b}\right\|_2^2](http://latex.codecogs.com/gif.latex?%5COmega%28%5Ctheta%29%3D%5Cfrac%7B1%7D%7B2%7D%5Cleft%5C%7C%5Cmathbf%7Bw-b%7D%5Cright%5C%7C_2%5E2)

where ![\mathbf{b}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bb%7D) is the arbitrary point and is the origin (ie. ![\mathbf{0}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B0%7D)) by default.

`page 225` Refer to Chapter 2 notes and Figure 2.3 for eigenvectors and scaling.

`page 226` Just for clarification, in Figure 7.1, the *first dimension* and ![w_1](http://latex.codecogs.com/gif.latex?w_1) refer to the horizontal x-axis.

In the first dimension, the eigenvalue of the Hessian of ![J](http://latex.codecogs.com/gif.latex?J) is small and this is illustrated by the solid contours being stretched along the horizontal axis.

Since the eigenvalue of the Hessian of ![J](http://latex.codecogs.com/gif.latex?J) is small for the first dimension, this means that changes to ![w_1](http://latex.codecogs.com/gif.latex?w_1) will only result in small changes to ![J](http://latex.codecogs.com/gif.latex?J).

Recall that the contour lines indicate regions with the same loss value ![J](http://latex.codecogs.com/gif.latex?J). Consider if we allow ![J](http://latex.codecogs.com/gif.latex?J) to increase by ![m](http://latex.codecogs.com/gif.latex?m) by adjusting either ![w_1](http://latex.codecogs.com/gif.latex?w_1) or ![w_2](http://latex.codecogs.com/gif.latex?w_2) (not both). We must adjust ![w_1](http://latex.codecogs.com/gif.latex?w_1) by a larger amount to achieve the same increase by ![m](http://latex.codecogs.com/gif.latex?m), as compared to adjusting ![w_2](http://latex.codecogs.com/gif.latex?w_2). 

This results in the solid contours being stretched along the horizontal ![w_1](http://latex.codecogs.com/gif.latex?w_1) axis.

On an additional note, the equilibrium ![\mathbf{\tilde{w}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Ctilde%7Bw%7D%7D) depicted in the graph is also determined by the regularization hyperparameter ![\alpha](http://latex.codecogs.com/gif.latex?%5Calpha).

`page 227` Interesting note on seeing regularization as increasing the *perceived* variance of the input.

