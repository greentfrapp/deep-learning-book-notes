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

`page 502` The corrupted sample can be as simple as adding random noise to the original sample.

`page 503` Equation 14.15 can be understood better in the context of Figures 14.4 and 14.5. The gradient field described by the equation shows the direction that ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Cx%7D) should move in order to increase ![\log p(\mathbf{x})](http://latex.codecogs.com/gif.latex?%5Clog%20p%28%5Cmathbf%7Bx%7D%29), log probability of ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Cx%7D). In simpler terms, the model learns how to change the modify the noisy sample into a clean sample, which has higher probability in the original data distribution.

`page 506`

> Some machine learning algorithms exploit this idea only insofar as they learn a function that behaves correctly on the manifold but that may have unusual behavior if given an input that is off the manifold.

This is an interesting note that seems to describe the current problem of adversarial samples for deep learning models ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)). These specially crafted adversarial samples might be considered *off the manifold*, since they do not occur *naturally* and do not appear in typical training sets. 

`page 506`

> The fact that ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Cx%7D) is drawn from the training data is crucial, because it means the autoencoder need not successfully reconstruct inputs that are not probable under the data-generating distribution.

Several things of note:

- As mentioned earlier in the text, an autoencoder that simply reconstructs every input perfectly is essentially an identity function and is typically overcomplete and pretty useless
- Researchers have exploited this concept to use autoencoders in anomaly detection ie. an autoencoder trained on just normal training data is unlikely to successfully reconstruct inputs outside of the normal data distribution (anomalies) hence anomalies can be detecteed by their higher reconstruction loss
- The training data is usually a tiny subset of the data distribution, which means that we have to consider how to ensure that the autoencoder learns the underlying data distribution and not just the empirical distribution, even though it is only exposed to the latter

`page 508` In Figure 14.7, it might be clearer if we see that the manifold structure is the set of ![\{x_0, x_1, x_2\}](http://latex.codecogs.com/gif.latex?%5C%7Bx_0%2C%20x_1%2C%20x_2%5C%7D) indicated by the vertical dashed lines.

The derivative of ![r(x)](http://latex.codecogs.com/gif.latex?r%28x%29) is small (0) around the data points, while being large (![\infty](http://latex.codecogs.com/gif.latex?%5Cinfty)) in the middle between two data points.

`page 510` The regularization imposed by Equation 14.18 'encourages the derivatives of f to be as small as possible'. This means that when the input is changed slightly, the encoding should not change by much.

`page 512`

> To clarify, the CAE is contractive only locally [...]

Given a sample, the CAE maps small perturbations of this sample to a similar region in the encoding space (since we want the encoding to remain roughly the same even when the input is perturbed slightly). But given two separate samples, the CAE might map these two samples to very different spaces.

`page 513` In my hardcopy of the textbook, Figure 14.10 has an errata, where the tangent vector samples for Local PCA and CAE should be swapped. This is not mentioned in the [errata list](https://docs.google.com/document/d/1ABlp7FluwZ0B82_fjNOFVQ2uOZkfuF8elbofhZmNXag/edit), but has been corrected in the [web version](http://www.deeplearningbook.org/contents/autoencoders.html) (see page 520).

`page 514` Equation 14.19 might appear confusing because the last term should be zero in the context of previous autoencoders. However, I believe in the case of PSD, ![\mathbf{h}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D) is a learned parameter during training (but not used in testing as mentioned in the text). Then we can use Equation 14.19 as the loss function to perform gradient descent on ![\mathbf{h}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D) and the parameters of ![f](http://latex.codecogs.com/gif.latex?f) and ![g](http://latex.codecogs.com/gif.latex?g).

`page 515`

> One trick that can accomplish this is simply to inject additive noise just before the sigmoid nonlinearity during training.

Note to self: this is an interesting trick for saturating activation functions that might prove useful.
