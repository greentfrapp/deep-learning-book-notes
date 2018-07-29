# 3 Probability and Information Theory

`page 53` Interesting and subtle distinction made between frequentist probability and Bayesian probability. 

Given two statements:

> A coin flip has a 50% chance of landing on heads.

> I have a 50% chance of passing my interview.

The first statement implies that if we repeat the coin flip infinite times, the coin will land on heads 50% of the time. But the second statement does not really imply that if I take the interview infinite times, I will pass it 50% of the time. In fact, the eventual outcome will be the same every time. 

In a way, the two statements are similar in that they both imply that, at the time of making the statement, I am completely uncertain about the binary outcome. However, the underlying meanings are subtly different.

`page 54` In the running for my favourite sentence in this book:

> A random variable is a variable that can take on different values randomly.

`page 56` The uniform distribution example for PDF took me awhile to understand. The part that caught me was that ![u(x;a,b)=\frac{1}{b-a}](http://latex.codecogs.com/gif.latex?u%28x%3Ba%2Cb%29%3D%5Cfrac%7B1%7D%7Bb-a%7D) gives the PDF rather than the probability. 

So, as a concrete example, to evaluate the probability of ![x](http://latex.codecogs.com/gif.latex?x) being in the interval ![[a,\frac{a+b}{2}]](http://latex.codecogs.com/gif.latex?%5Ba%2C%5Cfrac%7Ba&plus;b%7D%7B2%7D%5D), we have to calculate the integral ![\int_a^{\frac{a+b}{2}}\frac{1}{b-a}dx](http://latex.codecogs.com/gif.latex?%5Cint_a%5E%7B%5Cfrac%7Ba&plus;b%7D%7B2%7D%7D%5Cfrac%7B1%7D%7Bb-a%7Ddx), which gives us 0.5.

`page 57` Cool anecdote of the etymology of 'marginal probability'.

`page 57` Having just finished The Book of Why (Pearl & Mackenzie, 2018), I realize the below phrasing can be somewhat misleading.

> In many cases, we are interested in the probability of some event, given that some other event has happened. This is called a conditional probability.

The following paragraph actually addresses this. Put simply the conditional probability does not imply causation. To take a common example from Pearl & Mackenzie, the probability of the sun rising, given that the rooster has crowed, is high. But in no way implies that the rooster's crowing causes the sun to rise. 

More precisely, the conditional probability holds under the assumption of observation without intervention. If we observe the rooster crowing without having intervened and forcing it to crow, there is a good chance that the sun has risen. But if we intervene by forcing the rooster to crow, the crowing is no longer related to the sunrise.

`page 59` Nice example illustrating how independence is distinct from covariance - specifically how independence includes non-linear independence, while zero covariance only implies linear independence.

`page 63` 

> We often fix the covariance matrix to be a diagonal matrix. 

This would constrain the distribution such that the variables are modeled as linearly independent.

`page 64` Interesting notes on interpretations the empirical distribution (composed of Dirac distributions with a peak at every sample), in the context of deep learning.

> We can view the empirical distribution formed from a dataset of training examples as specifying the distribution that we sample from when we train a model on this dataset. Another important perspective on the empirical distribution is that it is the probability density that maximizes the likelihood of the training data.

`page 65` Notes on prior and posterior probabilities.

- Prior probability - The model's beliefs prior to observing the data
- Posterior probability - A conditional probability representing the model's beliefs after observing/given the data

`page 70` A nice example that builds on the previous chapter's notes on determinants.

We see that we have to take into account the change in volume and space as we convert from ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D) to ![\mathbf{y}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D). The resulting solution in Equation 3.47 uses the determinant of the Jacobian matrix. The Jacobian is simply the derivative of the variables in ![\mathbf{y}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D) with respect to the variables in ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D) (this is slightly different from the notation in the book - the book describes the general notation while here we describe it in context of the problem). 

The determinant of the Jacobian matrix then represents the volume change as we transform from ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D) space to ![\mathbf{y}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D) space.

`page 71` I like how Equation 3.48 (![I(x)=-\text{log}P(x)](http://latex.codecogs.com/gif.latex?I%28x%29%3D-%5Ctext%7Blog%7DP%28x%29)) fulfills all of the three properties in quantifying information.

- Likely events should have low information content and events guaranteed to happen should have no information content

If ![P(x)=1](http://latex.codecogs.com/gif.latex?P%28x%29%3D1) then ![I(x)=0](http://latex.codecogs.com/gif.latex?I%28x%29%3D0).

- Less likely events should have high information content

The negative log likelihood means that lower probabilities result in higher positive information values.

- Independent events should have additive information - a random event happening twice should convey twice as much information as the event happening once

If ![P(x)=P(\text{event})P(\text{event})](http://latex.codecogs.com/gif.latex?P%28x%29%3DP%28%5Ctext%7Bevent%7D%29P%28%5Ctext%7Bevent%7D%29) then ![I(x)=I(\text{event})+I(\text{event})](http://latex.codecogs.com/gif.latex?I%28x%29%3DI%28%5Ctext%7Bevent%7D%29&plus;I%28%5Ctext%7Bevent%7D%29).

`page 73` We can also understand Figure 3.6 in the context of KL divergence being the extra information required to convey distribution A using a code optimized for distribution B.

The left graph shows how ![q](http://latex.codecogs.com/gif.latex?q) should be distributed if we minimize ![D_{KL}(p\parallel q)](http://latex.codecogs.com/gif.latex?D_%7BKL%7D%28p%5Cparallel%20q%29). This can be seen as how we can distribute ![q](http://latex.codecogs.com/gif.latex?q) such that we can use the same 'code' for ![p](http://latex.codecogs.com/gif.latex?p) with minimal extra cost. We see that ![p](http://latex.codecogs.com/gif.latex?p) has two high-probability modes, which means that the optimal distribution for ![q](http://latex.codecogs.com/gif.latex?q) needs to be evenly spread out between these two modes as well.

The right graph shows how ![q](http://latex.codecogs.com/gif.latex?q) should be distributed if we minimize ![D_{KL}(q\parallel p)](http://latex.codecogs.com/gif.latex?D_%7BKL%7D%28q%5Cparallel%20p%29). This can be seen as how we can distribute ![q](http://latex.codecogs.com/gif.latex?q) such that we can convey information using the 'code' from ![p](http://latex.codecogs.com/gif.latex?p) with minimal adaptation needed. In that sense, we should align ![q](http://latex.codecogs.com/gif.latex?q) with one of the two modes that are already present in ![p](http://latex.codecogs.com/gif.latex?p) so that we can easily reuse the same 'code'.