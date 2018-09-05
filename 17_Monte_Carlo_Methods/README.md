# 17 Monte Carlo Methods

`page 581`

> Randomized algorithms fall into two rough categories: Las Vegas algorithms and Monte Carlo algorithms.

The Wikipedia [page](https://en.wikipedia.org/wiki/Las_Vegas_algorithm) for Las Vega algorithms has a nice comparison, which I quote here:

> Las Vegas algorithms can be contrasted with Monte Carlo algorithms, in which the resources used are bounded but the answer may be incorrect with a certain (typically small) probability. By an application of Markov's inequality, a Las Vegas algorithm can be converted into a Monte Carlo algorithm via early termination.

For the uninitiated, Monte Carlo algorithms typically involve using random samples to approximate an unknown value (that might be intractable or difficult to solve for). The Wikipedia [page](https://en.wikipedia.org/wiki/Monte_Carlo_method) for Monte Carlo algorithms has a nice demonstration of using a Monte Carlo method to approximate the value for ![\pi](http://latex.codecogs.com/gif.latex?%5Cpi).

![Using a Monte Carlo method to approximate pi](https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Pi_30K.gif/220px-Pi_30K.gif)

`page 584` For Equations 17.9 and 17.10, note that there is a difference in the subscript of the summation term - in 17.9 we sample from p while in 17.10 we sample from q, as per Equation 17.8.

Here is a more concrete but trivial example to demonstrate the transformation of the sampling estimator.

Suppose we have a coin flipping game where we gain 1 point if the coin lands heads-up and 0 point if the coin land tails-up. We want to find ![s](http://latex.codecogs.com/gif.latex?s), the expected value of the score. Following Equation 17.9, ![f(\mathbf{x})=1](http://latex.codecogs.com/gif.latex?f%28%5Cmathbf%7Bx%7D%29%3D1) if heads and ![f(\mathbf{x})=0](http://latex.codecogs.com/gif.latex?f%28%5Cmathbf%7Bx%7D%29%3D0) if tails. ![p(\mathbf{x})](http://latex.codecogs.com/gif.latex?p%28%5Cmathbf%7Bx%7D%29) is the distribution we sample from - in this case it is the coin flipping, which has 50% chance of landing on heads and 50% chance of tails. Of course, in most cases, we do not know ![p(\mathbf{x})](http://latex.codecogs.com/gif.latex?p%28%5Cmathbf%7Bx%7D%29) and/or ![f(\mathbf{x})=1](http://latex.codecogs.com/gif.latex?f%28%5Cmathbf%7Bx%7D%29%3D1), which makes the expectation difficult or intractable to evaluate.

According to Equation 17.9, we just have to flip the coin ![n](http://latex.codecogs.com/gif.latex?n) times and take the mean of our score in order to approximate the expected value. Suppose we have ![n=1000](http://latex.codecogs.com/gif.latex?n%3D1000), we might flip the coin 1000 times and find that we have 503 heads and 497 tails, which means that we have a total of 503 points. This is divided by [n](http://latex.codecogs.com/gif.latex?n) to give 0.503, which is our approximation of the expected score.

Next, suppose we have a biased coin and we know that ![\frac{p(\mathbf{x})}{q(\mathbf{x})}=\frac{2}{3}](http://latex.codecogs.com/gif.latex?%5Cfrac%7Bp%28%5Cmathbf%7Bx%7D%29%7D%7Bq%28%5Cmathbf%7Bx%7D%29%7D%3D%5Cfrac%7B2%7D%7B3%7D) for heads and ![\frac{p(\mathbf{x})}{q(\mathbf{x})}=2](http://latex.codecogs.com/gif.latex?%5Cfrac%7Bp%28%5Cmathbf%7Bx%7D%29%7D%7Bq%28%5Cmathbf%7Bx%7D%29%7D%3D%5Cfrac%7B2%7D%7B3%7D) for tails. In other words, ![q(\mathbf{x})=0.75](http://latex.codecogs.com/gif.latex?q%28%5Cmathbf%7Bx%7D%29%3D0.75) for heads and ![q(\mathbf{x})=0.25](http://latex.codecogs.com/gif.latex?q%28%5Cmathbf%7Bx%7D%29%3D0.25) for tails. Knowing the former relation between ![p(\mathbf{x})](http://latex.codecogs.com/gif.latex?p%28%5Cmathbf%7Bx%7D%29) and ![q(\mathbf{x})](http://latex.codecogs.com/gif.latex?q%28%5Cmathbf%7Bx%7D%29), we can substitute the relation into Equation 17.10 to use the biased coin to give an approximation for the expected score of the original unbiased coin. Again, suppose we have ![n=1000](http://latex.codecogs.com/gif.latex?n%3D1000), we might flip the biased coin 1000 times and find that we have 756 heads and 244 tails. For each heads, ![\frac{p(\mathbf{x})f(\mathbf{x})}{q(\mathbf{x})}=\frac{2}{3}](http://latex.codecogs.com/gif.latex?%5Cfrac%7Bp%28%5Cmathbf%7Bx%7D%29f%28%5Cmathbf%7Bx%7D%29%7D%7Bq%28%5Cmathbf%7Bx%7D%29%7D%3D%5Cfrac%7B2%7D%7B3%7D). For each tails, ![\frac{p(\mathbf{x})f(\mathbf{x})}{q(\mathbf{x})}=0](http://latex.codecogs.com/gif.latex?%5Cfrac%7Bp%28%5Cmathbf%7Bx%7D%29f%28%5Cmathbf%7Bx%7D%29%7D%7Bq%28%5Cmathbf%7Bx%7D%29%7D%3D0) (since ![f(\mathbf{x})=0](http://latex.codecogs.com/gif.latex?f%28%5Cmathbf%7Bx%7D%29%3D0)). Then the approximation for the expected value is given as 0.504.

`page 584` For Equation 17.13, notice the absolute sign on the ![f(\mathbf{x})](http://latex.codecogs.com/gif.latex?f%28%5Cmathbf%7Bx%7D%29) term.

> [...] where Z is the normalization constant, chosen so that ![q^\ast(\mathbf{x})](http://latex.codecogs.com/gif.latex?q%5E%5Cast%28%5Cmathbf%7Bx%7D%29) sums or integrates to 1 as appropriate.

Using the coin flipping game as an example, if heads gives a score of 2 and tails gives a score of -2 and we are using a fair coin, then ![Z=0.5(2)+0.5(|-2|)=2](http://latex.codecogs.com/gif.latex?Z%3D0.5%282%29&plus;0.5%28%7C-2%7C%29%3D2). 

`page 586` **Monte Carlo Markov Chain (MCMC).** The text actually provides a nice mathematical description, but here are some links for a more layman description (see first answer [here](https://stats.stackexchange.com/questions/165/how-would-you-explain-markov-chain-monte-carlo-mcmc-to-a-layperson))(see section titled 'Example: In-class test' [here](https://link.springer.com/article/10.3758/s13423-016-1015-8)).

`page 592`

> This means that Gibbs sampling will only very rarely flip the signs of these variables.

Suppose we begin with a = -1 and b = -1 and we want to update a. Since the probability of assigning a = -1 is close to 1, there is a near 100% chance of updating a to -1 again, instead of flipping it to +1. Likewise for b. Hence, there is a very low chance that we can move from our initial state of a = -1 and b = -1 to a = 1 and b = 1.

`page 593` There is an errata in my hardcopy for Figure 17.2, where the figures should be swapped. This has been mentioned in the [errata list](https://docs.google.com/document/d/1ABlp7FluwZ0B82_fjNOFVQ2uOZkfuF8elbofhZmNXag/edit) and updated in the web version for [Chapter 17](http://www.deeplearningbook.org/contents/monte_carlo.html) (see page 600).

`page 593` With reference to Figure 17.2, while the GAN does not have the mixing problem, it has been found that it suffers from a similar problem known as **mode collapse**, where a trained GAN model generates very similar images with little diversity. Recent works have attempted to address the mode collapse problem, such as the Wasserstein GAN (WGAN) by Arjovsky et al. ([2017](https://arxiv.org/abs/1701.07875)).

`page 594` **Temperature.** The concept of temperature has also been applied to the softmax function, when the output of the softmax is a probability distribution to be sampled from (eg. language model). In the case of softmax, the pre-softmaxed logits are divided by the temperature parameter ie. higher temperature gives lower resulting logits and a more uniform distribution after the softmax. See the Reinforcement Learning section of the Wikipedia [article](https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning) for the softmax. It is also a good exercise to try working out how the resulting distribution changes with temperature.
