# 6 Deep Feedforward Networks

`page 163`

> Deep feedforward networks, also called feedforward neural networks, or multilayer perceptrons (MLPs) [...]

Feedforward networks go by many names, but they all refer to the same notion of nested functions, where each function is a linear transformation of the input and optionally followed by a nonlinear activation function. 

Common names include:

- Deep Feedforward Networks (DFN)
- Feedforward Neural Networks (FNN)
- Feedforward Networks (FFN)
- Multilayer Perceptrons (MLP)
- Fully Connected Networks (FCN)
- Dense Networks

`page 165` ToDo: Implement linear and nonlinear algorithms for XOR

`page 174`

> [...] then it becomes possible to assign extremely high density to the correct training set outputs, resulting in cross-entropy approaching negative infinity.

To be honest I'm a bit confused here, since I thought cross-entropy should be more than zero (see [here](https://www.quora.com/What-is-the-range-of-the-cross-entropy-loss)). But the preceding line actually mentions that we are considering "real-valued output variables" ie. continuous. Since we are only considering minimizing the cross-entropy over the training samples, the cross-entropy can be reduced to a negative log conditional likelihood (see [here](https://stats.stackexchange.com/questions/215477/cross-entropy-equivalent-loss-suitable-for-real-valued-labels)). We can then refer to Equation 5.65 on page 130 for the log likelihood of a Gaussian output distribution. NOTE that the equation refers to log likelihood. Instead, we want to minimize negative log likelihood so the loss is more like:

![m\text{log}\sigma+\frac{m}{2}log(2\pi)+\sum_{i=1}^m\frac{\left\|\hat{y}^{(i)}-y^{(i)}\right\|^2}{2\sigma^2}](http://latex.codecogs.com/gif.latex?m%5Ctext%7Blog%7D%5Csigma&plus;%5Cfrac%7Bm%7D%7B2%7Dlog%282%5Cpi%29&plus;%5Csum_%7Bi%3D1%7D%5Em%5Cfrac%7B%5Cleft%5C%7C%5Chat%7By%7D%5E%7B%28i%29%7D-y%5E%7B%28i%29%7D%5Cright%5C%7C%5E2%7D%7B2%5Csigma%5E2%7D)

Basically negative Equation 5.65.

Now, if we allow the algorithm to optimize the variance/standard deviation as well, then it becomes possible for the algorithm to assign extremely low variance ~ 0, which makes the Gaussian almost like a Dirac delta function ie. density concentrated around the target label and zero everywhere else. This in turn causes the loss function to approach negative infinity (since we have to calculate log(~0)).


`page 175` ToDo: Visualize why mean squared/absolute errors lead to small gradients compared to cross-entropy

`page 176` **Linear Units for Gaussian Output Distributions.** For instance, the layer that generates the latent vectors for vanilla autoencoders commonly use linear units without any activation functions. This implicitly models the latent vector as the mean of a conditional (usually multivariate) Gaussian distribution (conditioned on the input sample to be encoded).

`page 177` **Sigmoid Units for Bernoulli Output Distributions.**

> If we begin with the assumption that the unnormalized log probabilities are linear in y and z [...]

Does the assumption make sense? Recall that the output is a Bernoulli distribution over a binary variable, which means ![y=0](http://latex.codecogs.com/gif.latex?y%3D0) or ![y=1](http://latex.codecogs.com/gif.latex?y%3D1). Then ![\log\tilde{P}(y)=yz](http://latex.codecogs.com/gif.latex?%5Clog%5Ctilde%7BP%7D%28y%29%3Dyz) simply means that ![\log\tilde{P}(y=0)=0](http://latex.codecogs.com/gif.latex?%5Clog%5Ctilde%7BP%7D%28y%3D0%29%3D0) or ![\log\tilde{P}(y=1)=z](http://latex.codecogs.com/gif.latex?%5Clog%5Ctilde%7BP%7D%28y%3D1%29%3Dz). Also recall that ![\tilde{P}(y)](http://latex.codecogs.com/gif.latex?%5Ctilde%7BP%7D%28y%29) refers to the unnormalized probability distribution. So it is okay that ![\tilde{P}(y=0)=1](http://latex.codecogs.com/gif.latex?%5Ctilde%7BP%7D%28y%3D0%29%3D1) and ![\tilde{P}(y=1)=\exp(z)](http://latex.codecogs.com/gif.latex?%5Ctilde%7BP%7D%28y%3D1%29%3D%5Cexp%28z%29), as long as we normalize it later. Setting ![\log\tilde{P}(y)=yz](http://latex.codecogs.com/gif.latex?%5Clog%5Ctilde%7BP%7D%28y%29%3Dyz) just means that we hold ![\log\tilde{P}(y=0)=0](http://latex.codecogs.com/gif.latex?%5Clog%5Ctilde%7BP%7D%28y%3D0%29%3D0) constant while allowing the algorithm to optimize the ![z](http://latex.codecogs.com/gif.latex?z) term in ![\log\tilde{P}(y=1)=z](http://latex.codecogs.com/gif.latex?%5Clog%5Ctilde%7BP%7D%28y%3D1%29%3Dz).

There is also a slight jump from Equation 6.22 to 6.23. Elaborated below:

![P(y)=\frac{\text{exp}(yz)}{\sum_{y'=0}^1\text{exp}(y'z)}](http://latex.codecogs.com/gif.latex?P%28y%29%3D%5Cfrac%7B%5Ctext%7Bexp%7D%28yz%29%7D%7B%5Csum_%7By%27%3D0%7D%5E1%5Ctext%7Bexp%7D%28y%27z%29%7D)

![P(y)=\frac{\text{exp}(yz)}{\text{exp}(0)+\text{exp}(z)}](http://latex.codecogs.com/gif.latex?P%28y%29%3D%5Cfrac%7B%5Ctext%7Bexp%7D%28yz%29%7D%7B%5Ctext%7Bexp%7D%280%29&plus;%5Ctext%7Bexp%7D%28z%29%7D)

![P(y)=\frac{\text{exp}(yz)}{1+\text{exp}(z)}](http://latex.codecogs.com/gif.latex?P%28y%29%3D%5Cfrac%7B%5Ctext%7Bexp%7D%28yz%29%7D%7B1&plus;%5Ctext%7Bexp%7D%28z%29%7D)

![\text{If } y=0, \text{ then }P(y)=\frac{1}{1+\text{exp}(z)}](http://latex.codecogs.com/gif.latex?%5Ctext%7BIf%20%7D%20y%3D0%2C%20%5Ctext%7B%20then%20%7DP%28y%29%3D%5Cfrac%7B1%7D%7B1&plus;%5Ctext%7Bexp%7D%28z%29%7D)

![\text{If } y=1, \text{ then }P(y)=\frac{\text{exp}(z)}{1+\text{exp}(z)}](http://latex.codecogs.com/gif.latex?%5Ctext%7BIf%20%7D%20y%3D1%2C%20%5Ctext%7B%20then%20%7DP%28y%29%3D%5Cfrac%7B%5Ctext%7Bexp%7D%28z%29%7D%7B1&plus;%5Ctext%7Bexp%7D%28z%29%7D)

This means we can also rewrite the equation as:

![P(y)=\frac{\text{exp}((2y-1)z)}{1+\text{exp}((2y-1)z)}](http://latex.codecogs.com/gif.latex?P%28y%29%3D%5Cfrac%7B%5Ctext%7Bexp%7D%28%282y-1%29z%29%7D%7B1&plus;%5Ctext%7Bexp%7D%28%282y-1%29z%29%7D)

without changing the possible values of ![P(y)](http://latex.codecogs.com/gif.latex?P%28y%29).

Finally, we can directly represent that as a sigmoidal transformation:

![P(y)=\sigma((2y-1)z)](http://latex.codecogs.com/gif.latex?P%28y%29%3D%5Csigma%28%282y-1%29z%29)

`page 178` On Equation 6.26, recall softplus from Equation 3.31 on page 66.

![\zeta(x)=\log(1+\exp(x))](http://latex.codecogs.com/gif.latex?%5Czeta%28x%29%3D%5Clog%281&plus;%5Cexp%28x%29%29)

![Softplus (green) and ReLU (blue) functions](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Rectifier_and_softplus_functions.svg/1200px-Rectifier_and_softplus_functions.svg.png)

*Softplus (green) and ReLU (blue) functions.*

Notice that the softplus (blue) function only saturates when the input is very negative. The proceeding paragraph in the text gives a good explanation of why this property of the softplus is great for gradient-based learning.

Also a side note, the sign of the input reverses when taking the negative log likelihood:

![-\log\sigma(x)=\zeta(-x)](http://latex.codecogs.com/gif.latex?-%5Clog%5Csigma%28x%29%3D%5Czeta%28-x%29)

Regarding numerical computation concerns, as the text suggests, the negative log likelihood can be written as a function of ![z](http://latex.codecogs.com/gif.latex?z) rather than of ![\sigma(z)](http://latex.codecogs.com/gif.latex?%5Csigma%28z%29), which yields the softplus function:

![J(\theta)=\log(1+\exp((1-2y)z))](http://latex.codecogs.com/gif.latex?J%28%5Ctheta%29%3D%5Clog%281&plus;%5Cexp%28%281-2y%29z%29%29)

However, this still yields an overflow problem when calculating the exponential term if ![(1-2y)z](http://latex.codecogs.com/gif.latex?%281-2y%29z) is too large. 

A possible solution is to realize that when ![(1-2y)z](http://latex.codecogs.com/gif.latex?%281-2y%29z) is large, 

![\log(1+\exp((1-2y)z))\approx(1-2y)z](http://latex.codecogs.com/gif.latex?%5Clog%281&plus;%5Cexp%28%281-2y%29z%29%29%5Capprox%281-2y%29z)

So we can simply add an `if` statement to filter out values of ![(1-2y)z](http://latex.codecogs.com/gif.latex?%281-2y%29z) beyond a certain threshold. In fact, as noted [here](https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python), setting the threshold at 30 already gives the approximate solution within an error of < 1e-10 in magnitude.

`page 181` Interesting note on how the typical approach for the softmax function actually overparameterizes the distribution. Although somewhat disappointingly, the restricted version apparently gives similar performance.

`page 182` **Heteroscedastic.** Refers to data distributions with subsets that have different variances/variabilities from each other.

`page 183` Nice notes on why using precision works better than variance in implementation, although the two concepts are interchangeable. 

`page 186`

> A function is differentiable at z only if both the left derivative and right derivative are defined and equal to each other.

A nice succinct definition, that precludes the ReLU due to the non-differentiable point at 0.

`page 187`

> When initializing the parameters of the affine transformation, it can be a good practice to set all elements of b to a small positive value, such as 0.1. Doing so makes it very likely that the rectified linear units will be initially active for most inputs in the training set and allow the derivatives to pass through.

`page 188` **Maxout units.** This seems to be less mainstream in current research works. Here's the [original paper](https://arxiv.org/abs/1302.4389) by Goodfellow et al. (2013) and the [paper](https://arxiv.org/abs/1312.6211) discussing catastrophic forgetting by Goodfellow et al. (2014).




