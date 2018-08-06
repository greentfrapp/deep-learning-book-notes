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

On that note, we can generalize the regularization term to regularize and push the parameters towards any arbitrary point with the following:

![\Omega(\theta)=\frac{1}{2}\left\|\mathbf{w-w}^{(o)}\right\|_2^2](http://latex.codecogs.com/gif.latex?%5COmega%28%5Ctheta%29%3D%5Cfrac%7B1%7D%7B2%7D%5Cleft%5C%7C%5Cmathbf%7Bw-w%7D%5E%7B%28o%29%7D%5Cright%5C%7C_2%5E2)

where ![\mathbf{w}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D%5E%7B%280%29%7D) is the arbitrary point and is assumed to be the origin (ie. ![\mathbf{0}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B0%7D)) by default. 

A similar approach can be taken for ![L^1](http://latex.codecogs.com/gif.latex?L%5E1) regularization, as seen in the footnore on page 227.

`page 225` Refer to Chapter 2 notes and Figure 2.3 for eigenvectors and scaling.

`page 226` Just for clarification, in Figure 7.1, the *first dimension* and ![w_1](http://latex.codecogs.com/gif.latex?w_1) refer to the horizontal x-axis.

In the first dimension, the eigenvalue of the Hessian of ![J](http://latex.codecogs.com/gif.latex?J) is small and this is illustrated by the solid contours being stretched along the horizontal axis.

Since the eigenvalue of the Hessian of ![J](http://latex.codecogs.com/gif.latex?J) is small for the first dimension, this means that changes to ![w_1](http://latex.codecogs.com/gif.latex?w_1) will only result in small changes to ![J](http://latex.codecogs.com/gif.latex?J).

Recall that the contour lines indicate regions with the same loss value ![J](http://latex.codecogs.com/gif.latex?J). Consider if we allow ![J](http://latex.codecogs.com/gif.latex?J) to increase by ![m](http://latex.codecogs.com/gif.latex?m) by adjusting either ![w_1](http://latex.codecogs.com/gif.latex?w_1) or ![w_2](http://latex.codecogs.com/gif.latex?w_2) (not both). We must adjust ![w_1](http://latex.codecogs.com/gif.latex?w_1) by a larger amount to achieve the same increase by ![m](http://latex.codecogs.com/gif.latex?m), as compared to adjusting ![w_2](http://latex.codecogs.com/gif.latex?w_2). 

This results in the solid contours being stretched along the horizontal ![w_1](http://latex.codecogs.com/gif.latex?w_1) axis.

On an additional note, the equilibrium ![\mathbf{\tilde{w}}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Ctilde%7Bw%7D%7D) depicted in the graph is also determined by the regularization hyperparameter ![\alpha](http://latex.codecogs.com/gif.latex?%5Calpha).

`page 227` Interesting note on seeing regularization as increasing the *perceived* variance of the input.

`page 228` For Equation 7.20, the first term of the equation (![\alpha\text{sign}(\mathbf{w})](http://latex.codecogs.com/gif.latex?%5Calpha%5Ctext%7Bsign%7D%28%5Cmathbf%7Bw%7D%29)) is the derivative of ![\alpha\left\|\mathbf{w}\right\|_1](http://latex.codecogs.com/gif.latex?%5Calpha%5Cleft%5C%7C%5Cmathbf%7Bw%7D%5Cright%5C%7C_1). The ![\text{sign}](http://latex.codecogs.com/gif.latex?%5Ctext%7Bsign%7D) part is because we are considering the magnitude/absolute value of ![\mathbf{w}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D) in the original loss equation, where ![|w_i|=w_i](http://latex.codecogs.com/gif.latex?%7Cw_i%7C%3Dw_i) if ![w_i\geq0](http://latex.codecogs.com/gif.latex?w_i%5Cgeq0) and ![|w_i|=-w_i](http://latex.codecogs.com/gif.latex?%7Cw_i%7C%3D-w_i) if ![w_i<0](http://latex.codecogs.com/gif.latex?w_i%3C0).

`page 229` One way to think about the two possible outcomes is to go back to the quadratic approximation of the loss function (Equation 7.22). We see that by setting ![w_i=w_i^*](http://latex.codecogs.com/gif.latex?w_i%3Dw_i%5E*), we cancel out the term with the Hessian (![\frac{1}{2}H_{i,i}(w_i^*-w_i^*)^2](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7DH_%7Bi%2Ci%7D%28w_i%5E*-w_i%5E*%29%5E2)) and we are left with the weight decay term ![\alpha|w_i^*|](http://latex.codecogs.com/gif.latex?%5Calpha%7Cw_i%5E*%7C).

If we then set ![w_i=0](http://latex.codecogs.com/gif.latex?w_i%3D0) instead, then the loss due to the Hessian term increases by ![\frac{1}{2}H_{i,i}(w_i^*)^2](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7DH_%7Bi%2Ci%7D%28w_i%5E*%29%5E2) while the loss due to the weight decay reduces by ![\alpha|w_i^*|](http://latex.codecogs.com/gif.latex?%5Calpha%7Cw_i%5E*%7C).

The overall change in loss is given as ![\frac{1}{2}H_{i,i}(w_i^*)^2-\alpha|w_i^*|](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7DH_%7Bi%2Ci%7D%28w_i%5E*%29%5E2-%5Calpha%7Cw_i%5E*%7C).

Suppose ![\frac{1}{2}H_{i,i}(w_i^*)^2-\alpha|w_i^*|\leq0](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7DH_%7Bi%2Ci%7D%28w_i%5E*%29%5E2-%5Calpha%7Cw_i%5E*%7C%5Cleq0), this means that the resulting change in loss is negative ie. we reduce the loss, which is what we want. In that case, we should definitely set ![w_i=0](http://latex.codecogs.com/gif.latex?w_i%3D0). Assuming ![w_i^*>0](http://latex.codecogs.com/gif.latex?w_i%5E*%3E0), some simple manipulation of ![\frac{1}{2}H_{i,i}(w_i^*)^2-\alpha|w_i^*|\leq0](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7DH_%7Bi%2Ci%7D%28w_i%5E*%29%5E2-%5Calpha%7Cw_i%5E*%7C%5Cleq0) will result in ![w_i^*\leq\frac{\alpha}{H_{i,i}}](http://latex.codecogs.com/gif.latex?w_i%5E*%5Cleq%5Cfrac%7B%5Calpha%7D%7BH_%7Bi%2Ci%7D%7D), which is the condition stated in case 1.

Case 2 can also be interpreted in a similar manner. Likewise for ![w_i^*<0](http://latex.codecogs.com/gif.latex?w_i%5E*%3C0).

`page 229` Good exercise to try to derive Equation 7.24 from 3.26. Note that the ![\gamma](http://latex.codecogs.com/gif.latex?%5Cgamma) term in Equation 3.26 is set to ![\frac{1}{\alpha}](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B%5Calpha%7D) in Equation 7.24. This also means that a smaller ![\alpha](http://latex.codecogs.com/gif.latex?%5Calpha) implies a wider and less peaky/sharp distribution.


