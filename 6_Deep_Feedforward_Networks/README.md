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

`page 193`

> This is easiest to see in the binary case: the number of possible binary functions on vectors ![\mathbf{v}\in\{0,1\}^n](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bv%7D%5Cin%5C%7B0%2C1%5C%7D%5En) is ![2^{2^n}](http://latex.codecogs.com/gif.latex?2%5E%7B2%5En%7D) and selecting one such function requires ![2^n](http://latex.codecogs.com/gif.latex?2%5En) bits, which will in general require ![O(2^n)](http://latex.codecogs.com/gif.latex?O%282%5En%29) degrees of freedom.

Let's consider ![\mathbf{v}\in\{0,1\}^2](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bv%7D%5Cin%5C%7B0%2C1%5C%7D%5E2). This means that ![\mathbf{v}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bv%7D) contains vectors that are 2-dimensional and each dimension's value can only take on either 0 or 1. We can enumerate the entire set of ![\mathbf{v}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bv%7D) as:

![\begin{bmatrix}0\\0\end{bmatrix},\begin{bmatrix}0\\1\end{bmatrix},\begin{bmatrix}1\\0\end{bmatrix},\begin{bmatrix}1\\1\end{bmatrix}](http://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D0%5C%5C0%5Cend%7Bbmatrix%7D%2C%5Cbegin%7Bbmatrix%7D0%5C%5C1%5Cend%7Bbmatrix%7D%2C%5Cbegin%7Bbmatrix%7D1%5C%5C0%5Cend%7Bbmatrix%7D%2C%5Cbegin%7Bbmatrix%7D1%5C%5C1%5Cend%7Bbmatrix%7D)

Next we consider the number of possible binary functions on this set of vectors. Simply consider a function as a mapping between every possible input vector and its output. In our case, since we are considering only binary functions, the only possible outputs are 0 or 1.

Here's one possible function/mapping:

![f\left(\begin{bmatrix}0\\0\end{bmatrix}\right)=0](http://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7D0%5C%5C0%5Cend%7Bbmatrix%7D%5Cright%29%3D0)
![f\left(\begin{bmatrix}0\\1\end{bmatrix}\right)=0](http://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7D0%5C%5C1%5Cend%7Bbmatrix%7D%5Cright%29%3D0)
![f\left(\begin{bmatrix}1\\0\end{bmatrix}\right)=0](http://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7D1%5C%5C0%5Cend%7Bbmatrix%7D%5Cright%29%3D0)
![f\left(\begin{bmatrix}1\\1\end{bmatrix}\right)=0](http://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7D1%5C%5C1%5Cend%7Bbmatrix%7D%5Cright%29%3D0)

Here the function maps all the inputs to 0.

Here's another possible function/mapping:

![f\left(\begin{bmatrix}0\\0\end{bmatrix}\right)=0](http://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7D0%5C%5C0%5Cend%7Bbmatrix%7D%5Cright%29%3D0)
![f\left(\begin{bmatrix}0\\1\end{bmatrix}\right)=1](http://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7D0%5C%5C1%5Cend%7Bbmatrix%7D%5Cright%29%3D1)
![f\left(\begin{bmatrix}1\\0\end{bmatrix}\right)=1](http://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7D1%5C%5C0%5Cend%7Bbmatrix%7D%5Cright%29%3D1)
![f\left(\begin{bmatrix}1\\1\end{bmatrix}\right)=1](http://latex.codecogs.com/gif.latex?f%5Cleft%28%5Cbegin%7Bbmatrix%7D1%5C%5C1%5Cend%7Bbmatrix%7D%5Cright%29%3D1)

In this case, this function acts like an OR gate.

In general, we can see that there are ![2^k](http://latex.codecogs.com/gif.latex?2%5Ek) possible unique functions/mappings where ![k](http://latex.codecogs.com/gif.latex?k) refers to the number of possible inputs. Since we only consider binary values, ![k=2^n](http://latex.codecogs.com/gif.latex?k%3D2%5En). Which means the total number of possible functions is ![2^{2^n}](http://latex.codecogs.com/gif.latex?2%5E%7B2%5En%7D).

`page 194` For Figure 6.5, recall that the absolute value rectification unit is simply ![g(z)=\left|z\right|](http://latex.codecogs.com/gif.latex?g%28z%29%3D%5Cleft%7Cz%5Cright%7C) (see page 187). See original paper [here](https://arxiv.org/abs/1402.1869).

`page198`

> The back-propagation algorithm can be applied to these tasks as well and is not restricted to computing the gradient of the cost function with respect to the parameters.

Some common uses include computing the gradient with respect to the input, in order to craft [adversarial examples](https://arxiv.org/abs/1412.6572), as well as to maximize activation of certain nodes/neurons for [feature visualization](https://distill.pub/2017/feature-visualization/).

`page 200` Similar computational graphs can also be visualized with tools such as [Tensorboard](https://www.tensorflow.org/guide/graph_viz), which can be helpful for analysis and debugging.

`page 202` ![Pa(u^{(i)})](http://latex.codecogs.com/gif.latex?Pa%28u%5E%7B%28i%29%7D%29) refers to indices of parents of ![u^{(i)}](http://latex.codecogs.com/gif.latex?u%5E%7B%28i%29%7D).

`page 202` For Equation 6.49, ![\sum_{i:j\in Pa(u^{(i)})}](http://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3Aj%5Cin%20Pa%28u%5E%7B%28i%29%7D%29%7D) refers to a summation across all ![u^{(i)}](http://latex.codecogs.com/gif.latex?u%5E%7B%28i%29%7D), whose parents include node ![u^{(j)}](http://latex.codecogs.com/gif.latex?u%5E%7B%28j%29%7D). In other words, summation across all children of ![u^{(j)}](http://latex.codecogs.com/gif.latex?u%5E%7B%28j%29%7D).

For example, suppose we have an input node ![u^{(1)}](http://latex.codecogs.com/gif.latex?u%5E%7B%281%29%7D). This node is a parent of both nodes ![u^{(2)}](http://latex.codecogs.com/gif.latex?u%5E%7B%282%29%7D) and ![u^{(3)}](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D). And suppose nodes ![u^{(2)}](http://latex.codecogs.com/gif.latex?u%5E%7B%282%29%7D) and ![u^{(3)}](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D) are parents of ![u^{(4)}](http://latex.codecogs.com/gif.latex?u%5E%7B%284%29%7D).

Then Equation 6.49 tells us that:

![\frac{\partial u^{(4)}}{\partial u^{(1)}}=\frac{\partial u^{(4)}}{\partial u^{(3)}}\frac{\partial u^{(3)}}{\partial u^{(1)}}+\frac{\partial u^{(4)}}{\partial u^{(2)}}\frac{\partial u^{(2)}}{\partial u^{(1)}}](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20u%5E%7B%284%29%7D%7D%7B%5Cpartial%20u%5E%7B%281%29%7D%7D%3D%5Cfrac%7B%5Cpartial%20u%5E%7B%284%29%7D%7D%7B%5Cpartial%20u%5E%7B%283%29%7D%7D%5Cfrac%7B%5Cpartial%20u%5E%7B%283%29%7D%7D%7B%5Cpartial%20u%5E%7B%281%29%7D%7D&plus;%5Cfrac%7B%5Cpartial%20u%5E%7B%284%29%7D%7D%7B%5Cpartial%20u%5E%7B%282%29%7D%7D%5Cfrac%7B%5Cpartial%20u%5E%7B%282%29%7D%7D%7B%5Cpartial%20u%5E%7B%281%29%7D%7D)

`page 202` For context, we can consider Algorithm 6.1 as computing the scalar output of a neural network.

In this neural network, there are a total of ![n](http://latex.codecogs.com/gif.latex?n) nodes, including input, hidden and output nodes. The input is a vector of ![n_i](http://latex.codecogs.com/gif.latex?n_i) dimensions, while the output is a scalar (1-dimensional).

As mentioned in the text:

> We will assume that the nodes of the graph have been ordered in such a way that we can compute their output one after the other, starting at ![u^{(n_i+1)}](http://latex.codecogs.com/gif.latex?u%5E%7B%28n_i&plus;1%29%7D) and going up to ![u^{(n)}](http://latex.codecogs.com/gif.latex?u%5E%7B%28n%29%7D).

As an example, suppose ![n_i=2](http://latex.codecogs.com/gif.latex?n_i%3D2) (the input vector is 2-dimensional) and the first hidden layer is 3-dimensional. This means that ![u^{(1)},u^{(2)}](http://latex.codecogs.com/gif.latex?u%5E%7B%281%29%7D%2Cu%5E%7B%282%29%7D) are the nodes for the input layer. And ![u^{(3)},u^{(4)},u^{(5)}](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D%2Cu%5E%7B%284%29%7D%2Cu%5E%7B%285%29%7D) are the nodes for the first hidden layer. The ordering of the indices is such that each node can be computed using the values of the previous nodes eg. node ![u^{(3)}](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D) can be computed using ![u^{(1)},u^{(2)}](http://latex.codecogs.com/gif.latex?u%5E%7B%281%29%7D%2Cu%5E%7B%282%29%7D).

The first `for` loop in Algorithm 6.1 simply assigns the input vector to the nodes of the input layer.

The second `for` loop computes the values of subsequent nodes in order of the indices. Continuing with the above example, suppose we want to compute the value for node ![u^{(3)}](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D). We first assign the parents of ![u^{(3)}](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D) to the set ![\mathbb{A}](http://latex.codecogs.com/gif.latex?%5Cmathbb%7BA%7D). For a regular feedforward network, the parents of each node are all the nodes in the immediate preceding layer eg. for node ![u^{(3)}](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D), its parents are ![u^{(1)},u^{(2)}](http://latex.codecogs.com/gif.latex?u%5E%7B%281%29%7D%2Cu%5E%7B%282%29%7D). So ![\mathbb{A}=\{u^{(1)},u^{(2)}\}](http://latex.codecogs.com/gif.latex?%5Cmathbb%7BA%7D%3D%5C%7Bu%5E%7B%281%29%7D%2Cu%5E%7B%282%29%7D%5C%7D). Then we apply the corresponding function for node ![u^{(3)}](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D) ie.

![u^{(3)}=f^{(3)}(\mathbb{A})=f^{(3)}(\{u^{(1)},u^{(2)}\})](http://latex.codecogs.com/gif.latex?u%5E%7B%283%29%7D%3Df%5E%7B%283%29%7D%28%5Cmathbb%7BA%7D%29%3Df%5E%7B%283%29%7D%28%5C%7Bu%5E%7B%281%29%7D%2Cu%5E%7B%282%29%7D%5C%7D%29)

We do this repeatedly until we arrive at the value for the output node ![u^{(n)}](http://latex.codecogs.com/gif.latex?u%5E%7B%28n%29%7D).

`page 204` Algorithm 6.2 (backpropagation) is actually similar to Algorithm 6.1, but executed in reverse order, where the derivatives for nodes are computed based on the derivatives of their children, in accordance with the chain rule. In this case, we focus on calculating the derivatives with respect to the node values. However, in actually implementation, we typically use backpropagation to also calculate the derivatives with respect to the weight matrices and biases, in order to perform updates via gradient descent. Refer to Algorithms 6.3 and 6.4 for forward and backward propagation through a neural network.

`page 207` **Symbol-to-number and Symbol-to-symbol Differentiation.** It might be difficult to appreciate the difference between the two approaches, especially given the following sentence in the next page:

> The description of the symbol-to-symbol based approach subsumes the symbol-to-number approach.

The difference is stated a few sentences later as:

> The key difference is that the symbol-to-number approach does not expose the graph.

Here's a nice symbol-to-symbol [example](https://stackoverflow.com/questions/44342432/is-gradient-in-the-tensorflows-graph-calculated-incorrectly) (see first answer) of the additional nodes constructed in Tensorflow for the derivatives. Specifically, we see that when we construct a cosine function and a gradient operator for calculating the gradient of the cosine function, the graph shows both `cos` and `sin` nodes, since the derivative of a cosine function is negative sine. The `sin` node is the additional node added for calculating the derivative.

On the other hand, the symbol-to-number approach does not explicitly construct the nodes for the derivatives. Instead the derivatives are calculating during runtime locally for each node/edge in the graph. See [here](https://github.com/attractivechaos/kann/blob/master/doc/02dev.md#automatic-differentiation-and-computational-graph) for an example.

Crucially, because the nodes for the derivative operations are not constructed in the symbol-to-number approach, it can be difficult to obtain derivatives of derivatives (ie. higher derivatives), which may be useful in cases such as meta-learning ([Finn et al., 2017](https://arxiv.org/abs/1703.03400)).

`page 209` Notice that gradient with respect to ![\mathbf{A}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BA%7D) is given by ![\mathbf{GB}^\top](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BGB%7D%5E%5Ctop), while gradient with respect to ![\mathbf{B}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BB%7D) is given by ![\mathbf{A^\top G}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BA%5E%5Ctop%20G%7D). This is due to the noncommutative properties of matrix multiplication and is easy to see why if we consider ![\mathbf{A}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BA%7D) and ![\mathbf{B}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BB%7D) to have different shapes.

`page 209`

> The `op.bprop` method should always pretend that all its inputs are distinct from each other, even if they are not.

Interesting note with a great example!

`page 210` Some clarifying notes on the `for` loop in Algorithm 6.6, which describes the `build_grad` function.

The `for` loop computes partial derivatives contributed by the children of **V** and we sum all of these to get the gradient for **V** (see Equation 6.49). The `for` loop does this by enumerating across all the children of **V** and doing the follow:

1. Get the operation (`sum`, `mul` etc.) leading to that child
2. Perform `build_grad` on the child to get the gradient of the output (typically loss) with respect to the child (this results in a recursive function)
3. Calculate partial derivative contributed by the child using `op.bprop`

`page 209` ToDo: Implement annotated backpropagation in NumPy for common operations.

`page 211` On a side note, **dynamic programming** is a popular paradigm that often uses recursive functions and memory to simplify algorithms and reduce runtime.

`page 215` Refer to [this](https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax) (take note that there are several small typos) for an explanation of why ![\frac{\partial J}{\partial z_i}=q_i-p_i](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20z_i%7D%3Dq_i-p_i).

`page 219` More precisely, Jarrett et al. ([2009](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)) observed that fixed random weights for feature extraction layers paired with a linear classifier layer that is trained in supervised mode is sufficient to achieve decent performance, as long as they include absolute value rectification and contrast normalization (refer to Sections 3 and 4 and Table 1 in the paper).
