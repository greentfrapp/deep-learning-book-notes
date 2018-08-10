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

`page 232` Nice easily-understood example of why regularization is helpful in the underdetermined case - logistic regression applied to linearly separable classes.

`page 233` Interesting interpretation relating pseudo-inverse to regularization.

`page 234` **Out-of-plane Rotation** Suppose you have a camera focused on a regular cup on a table, which gives you an image of the cup. If you rotate either the camera or the environment, such that the axis of rotation is perpendicular to the image surface, that is an in-plane rotation. Rotating along any other axis results in an out-of-plane rotation. 

For instance, we can visualize what it is like to turn the camera upside down (in-plane rotation) by simply flipping the image. However, in order to see the hidden/back sided of the cup, we will have to physically adjust the cup (out-of-plane rotation), since we cannot perform any transformation to the original image that will give us the desired effect.

`page 236` **Label Smoothing.** I wonder if there is a good rule-of-thumb value for ![\epsilon](http://latex.codecogs.com/gif.latex?%5Cepsilon), other than ![\epsilon>1/k](http://latex.codecogs.com/gif.latex?%5Cepsilon%3E1/k).

`page 238` **Multitask Learning.** 

Reinforcement learning often uses a similar idea, where the value network and the policy network shares the same parameters/low-level layers.

A related field is domain adaptation, which involves learning a classifier for a task without labels, using another task from a different but related domain that has labels.

This is also related to the concept of transfer learning, which is in turn related to distillation ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)).

`page 242` **Early Stopping.** As mentioned in the text, early stopping is an extremely simple form of regularization and modern libraries often have built-in functions or classes that facilitate early stopping (examples include [Keras](https://keras.io/callbacks/#earlystopping), [Tensorflow](https://www.tensorflow.org/versions/r1.1/get_started/monitors#configuring_a_validationmonitor_for_streaming_evaluation)) and [PyTorch](https://pytorch.org/ignite/handlers.html#ignite.handlers.EarlyStopping). **Note:** it is extremely important that a separate validation set be set aside for early stopping and not to use the test set. - same as for other forms of hyperparameter tuning.

`page 245` Again, with reference to chapter 2, ![\lambda_i](http://latex.codecogs.com/gif.latex?%5Clambda_i) refers to the eigenvalue of the corresponding eigenvector in ![\mathbf{Q}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D).

One way to understand Equation 7.40 is to first see that if we did not set ![\mathbf{w}^{(0)}=\mathbf{0}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%3D%5Cmathbf%7B0%7D), then the equation should be:

![\mathbf{Q}^\top\mathbf{w}^{(\tau)}=\mathbf{Q}^\top\mathbf{w}^{(0)}+[\mathbf{I}-(\mathbf{I}-\epsilon\mathbf{\Lambda})^\tau](\mathbf{Q}^\top\mathbf{w}^*-\mathbf{Q}^\top\mathbf{w}^{(0)})](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%28%5Ctau%29%7D%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D&plus;%5B%5Cmathbf%7BI%7D-%28%5Cmathbf%7BI%7D-%5Cepsilon%5Cmathbf%7B%5CLambda%7D%29%5E%5Ctau%5D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*-%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%29)

The first term in the left-hand side is the starting point ![\mathbf{Q}^\top\mathbf{w}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D). The second point represents the amount that we move towards the unregularized goal ![\mathbf{Q}^\top\mathbf{w}^*](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*).

Then by setting ![\mathbf{w}^{(0)}=\mathbf{0}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%3D%5Cmathbf%7B0%7D), we arrive back at Equation 7.40.

Now consider if we have 0 updates (![\tau=0](http://latex.codecogs.com/gif.latex?%5Ctau%3D0)). Then we have:

![\mathbf{Q}^\top\mathbf{w}^{(0)}=[\mathbf{I}-(\mathbf{I}-\epsilon\mathbf{\Lambda})^0]\mathbf{Q}^\top\mathbf{w}^*=\mathbf{0}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%3D%5B%5Cmathbf%7BI%7D-%28%5Cmathbf%7BI%7D-%5Cepsilon%5Cmathbf%7B%5CLambda%7D%29%5E0%5D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*%3D%5Cmathbf%7B0%7D)

which makes sense since the text mentions that we set ![\mathbf{w}^{(0)}=\mathbf{0}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%3D%5Cmathbf%7B0%7D).

Using the more general form of the equation (see above), we will get the rather tautological equation ![\mathbf{Q}^\top\mathbf{w}^{(0)}=\mathbf{Q}^\top\mathbf{w}^{(0)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D).

Next after 1 update, we will move towards ![\mathbf{Q}^\top\mathbf{w}^*](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*). The amount that we move by is dictated by the eigenvalues of the Hessian ![\mathbf{H}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BH%7D) (gradient of ![\hat{J}](http://latex.codecogs.com/gif.latex?%5Chat%7BJ%7D)) and learning rate ![\epsilon](http://latex.codecogs.com/gif.latex?%5Cepsilon). In other words:

![\mathbf{Q}^\top\mathbf{w}^{(1)}=\mathbf{Q}^\top\mathbf{w}^{(0)}+[\mathbf{I}-(\mathbf{I}-\epsilon\mathbf{\Lambda})^1](\mathbf{Q}^\top\mathbf{w}^*-\mathbf{Q}^\top\mathbf{w}^{(0)})=\mathbf{Q}^\top\mathbf{w}^{(0)}+\epsilon\mathbf{\Lambda}(\mathbf{Q}^\top\mathbf{w}^*-\mathbf{Q}^\top\mathbf{w}^{(0)})](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%281%29%7D%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D&plus;%5B%5Cmathbf%7BI%7D-%28%5Cmathbf%7BI%7D-%5Cepsilon%5Cmathbf%7B%5CLambda%7D%29%5E1%5D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*-%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%29%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D&plus;%5Cepsilon%5Cmathbf%7B%5CLambda%7D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*-%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%29)

And again since ![\mathbf{w}^{(0)}=\mathbf{0}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%3D%5Cmathbf%7B0%7D), then we have ![\mathbf{Q}^\top\mathbf{w}^{(1)}=\epsilon\mathbf{\Lambda}\mathbf{Q}^\top\mathbf{w}^*](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%281%29%7D%3D%5Cepsilon%5Cmathbf%7B%5CLambda%7D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*).


In the second update, we will again move towards ![\mathbf{Q}^\top\mathbf{w}^*](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*), but we take this second step from ![\mathbf{Q}^\top\mathbf{w}^{(1)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%281%29%7D). This means:

![\mathbf{Q}^\top\mathbf{w}^{(2)}=\mathbf{Q}^\top\mathbf{w}^{(1)}+\epsilon\mathbf{\Lambda}(\mathbf{Q}^\top\mathbf{w}^*-\mathbf{Q}^\top\mathbf{w}^{(1)})](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%282%29%7D%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%281%29%7D&plus;%5Cepsilon%5Cmathbf%7B%5CLambda%7D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*-%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%281%29%7D%29)

It is easy to see that we can interpret this as a recursive function:

![\mathbf{Q}^\top\mathbf{w}^{(\tau)}=\mathbf{Q}^\top\mathbf{w}^{(\tau-1)}+\epsilon\mathbf{\Lambda}(\mathbf{Q}^\top\mathbf{w}^*-\mathbf{Q}^\top\mathbf{w}^{(\tau-1)})](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%28%5Ctau%29%7D%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%28%5Ctau-1%29%7D&plus;%5Cepsilon%5Cmathbf%7B%5CLambda%7D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*-%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%28%5Ctau-1%29%7D%29)

Furthermore, by setting ![\mathbf{w}^{(0)}=\mathbf{0}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bw%7D%5E%7B%280%29%7D%3D%5Cmathbf%7B0%7D), we get

![\mathbf{Q}^\top\mathbf{w}^{(2)}=\epsilon\mathbf{\Lambda}(\mathbf{Q}^\top\mathbf{w}^*)+\epsilon\mathbf{\Lambda}(\mathbf{Q}^\top\mathbf{w}^*-\epsilon\mathbf{\Lambda}(\mathbf{Q}^\top\mathbf{w}^*))](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%282%29%7D%3D%5Cepsilon%5Cmathbf%7B%5CLambda%7D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*%29&plus;%5Cepsilon%5Cmathbf%7B%5CLambda%7D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*-%5Cepsilon%5Cmathbf%7B%5CLambda%7D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*%29%29)

since ![\mathbf{Q}^\top\mathbf{w}^{(1)}=\epsilon\mathbf{\Lambda}(\mathbf{Q}^\top\mathbf{w}^*)](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%281%29%7D%3D%5Cepsilon%5Cmathbf%7B%5CLambda%7D%28%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*%29).

We can rearrange the terms to arrive at Equation 7.40.

![\mathbf{Q}^\top\mathbf{w}^{(2)}=\mathbf{Q}^\top\mathbf{w}^*(2\epsilon\mathbf{\Lambda}-\epsilon\mathbf{\Lambda}^2)=\mathbf{Q}^\top\mathbf{w}^*(\mathbf{I}-(\mathbf{I}-2\epsilon\mathbf{\Lambda}+\epsilon\mathbf{\Lambda}^2))=\mathbf{Q}^\top\mathbf{w}^*[\mathbf{I}-(\mathbf{I}-\epsilon\mathbf{\Lambda})^2]](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E%7B%282%29%7D%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*%282%5Cepsilon%5Cmathbf%7B%5CLambda%7D-%5Cepsilon%5Cmathbf%7B%5CLambda%7D%5E2%29%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*%28%5Cmathbf%7BI%7D-%28%5Cmathbf%7BI%7D-2%5Cepsilon%5Cmathbf%7B%5CLambda%7D&plus;%5Cepsilon%5Cmathbf%7B%5CLambda%7D%5E2%29%29%3D%5Cmathbf%7BQ%7D%5E%5Ctop%5Cmathbf%7Bw%7D%5E*%5B%5Cmathbf%7BI%7D-%28%5Cmathbf%7BI%7D-%5Cepsilon%5Cmathbf%7B%5CLambda%7D%29%5E2%5D)

`page 246` The approach proposed by Lasserre et al. ([2006](https://ieeexplore.ieee.org/document/1640745/)) is very much related to (and likely inspired) algorithms used in domain adaptation, including works by Long et al. ([2015](https://arxiv.org/abs/1502.02791)) and Ganin & Lempitsky ([2015](https://arxiv.org/abs/1409.7495)).

`page 247` While it is obvious once you think about it, it is still interesting to see convolutional networks as an example of parameter sharing. In the same way, RNNs can also be seen as sharing parameters across multiple steps of an input sequence.

`page 249` **Ensembled Methods.** As evidence of their effectiveness, ensemble methods are often used in Kaggle competitions and real-world implementations. Although important considerations are the memory requirements and inference/prediction speed, since we have to store and use several models. Ensemble methods are analogous to how multiple measurements are made with a ruler (physics lab in schools, anyone?) and we take the average of all the measurements in order to reduce uncertainty.

`page 250`

> on average around two-thirds of the examples from the original dataset are found in the resulting training set, if it has the same size as the original

This is an interesting observation that is explained [here](https://stats.stackexchange.com/questions/88980/why-on-average-does-each-bootstrap-sample-contain-roughly-two-thirds-of-observat) (see the second answer, then the first answer) and also relates to the .632 rule mentioned by Efron ([1983](https://www.jstor.org/stable/2288636)) and elaborated by Efron & Tibshirani ([1997](https://www.jstor.org/stable/2965703)).

`page 260` Interesting emphasis on the importance of the multiplicative property of dropout.

`page 261` An interesting note on how batch normalization can also have a regularizing effect, due to the additive and multiplicative noise in the normalization.

`page 262`

> Unfortunately, the value of a linear function can change very rapidly if it has numerous inputs.

This forms the basis of the Fast Gradient Sign Method (Goodfellow et al., [2014](https://arxiv.org/abs/1412.6572)), as well as other methods for generating adversarial samples with small perturbations.

`page 265` 

> Second, the infinitesimal approach poses difficulties for models based on rectified linear units. These models can only shrink their derivatives by turning units off or shrinking their weights. They are not able to shrink their derivatives by saturating at a high value with large weights, as sigmoid or tanh units can.

An interesting and important distinction between ReLU and sigmoid or tanh units.






