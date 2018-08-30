# 14 Autoencoders

`page 493`

> If an autoencoder succeeds in simply learning to set ![g(f(\mathbf{x}))=\mathbf{x}](http://latex.codecogs.com/gif.latex?g%28f%28%5Cmathbf%7Bx%7D%29%29%3D%5Cmathbf%7Bx%7D) everywhere, then it is not especially useful.

A simple example is if both g and f is the identity function ie. ![g(\mathbf{x})=f(\mathbf{x})=\mathbf{Ix}](http://latex.codecogs.com/gif.latex?g%28%5Cmathbf%7Bx%7D%29%3Df%28%5Cmathbf%7Bx%7D%29%3D%5Cmathbf%7BIx%7D), which is also pretty useless.

`page 495`

> Theoretically, one could imagine that an autoencoder with a one-dimensional code but a very powerful nonlinear encoder could learn to represent each training example ![\mathbf{x}^{(i)}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D%5E%7B%28i%29%7D) with the code ![i](http://latex.codecogs.com/gif.latex?i).

In this case, the autoencoder is just acting like a dictionary, mapping each training example to an index. The latent space spans a line and each training example is mapped to a 1D point on the line. If this happens, the resulting autoencoder is useless since the latent space tells us nothing except for the sequence of the training examples. In other words, the autoencoder acts just like a table/array storing the examples.

Note that in this case, the **overcapacity** is attributed to an excessively powerful encoder/decoder, even though the latent space is only 1D. In the **overcomplete** case, with reference to the text, the latent space is a higher dimension than the input space, which makes it trivial for the autoencoder to copy the input, such as with the identity function (see above).

`page 496` For Equation 14.2, note that we are imposing a preference towards sparse latent vectors, instead of sparse weights. This is also why there is no straightforward Bayesian interpretation, as explained in the text.

`page 496`

> In this view, regularized maximum likelihood corresponds to maximizing ![\log p(\mathbf{\theta\mid x})](http://latex.codecogs.com/gif.latex?%5Clog%20p%28%5Cmathbf%7B%5Ctheta%5Cmid%20x%7D%29), which is equivalent to maximizing ![\log p(\mathbf{x\mid\theta})+\log p(\mathbf{\theta})](http://latex.codecogs.com/gif.latex?%5Clog%20p%28%5Cmathbf%7Bx%5Cmid%5Ctheta%7D%29&plus;%5Clog%20p%28%5Cmathbf%7B%5Ctheta%7D%29).

This is simply using Bayes' Theorem and then removing ![-\log p(\mathbf{x})](http://latex.codecogs.com/gif.latex?-%5Clog%20p%28%5Cmathbf%7Bx%7D%29), which does not depend on ![\mathbf{\theta}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Ctheta%7D).

`page 497` In case Equation 14.3 is confusing, here is an analogy, which will be equivalent to the equation without the log operation. Suppose I flip a coin and roll a die. The probability of getting a heads for the coin flip is equal to the probability of getting a heads and rolling a one + the probability of getting a heads and rolling a two + ... + the probability of getting a heads and rolling a six.

`page 498`

> Denoising training forces ![f](http://latex.codecogs.com/gif.latex?f) and ![g](http://latex.codecogs.com/gif.latex?g) to implicitly learn the structure of ![p_\text{data}(\mathbf{x})](http://latex.codecogs.com/gif.latex?p_%5Ctext%7Bdata%7D%28%5Cmathbf%7Bx%7D%29) [...]

One factor for this is that there is no *structure* in random noise, which prevents encoding of the noise components.

`page 499` The gradient term in Equation 14.11 describes how much ![h_i](http://latex.codecogs.com/gif.latex?h_i) changes when ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D) changes.
