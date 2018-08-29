# 13 Linear Factor Models

`page 480` Here, a factorial distribution means that the components of ![\mathbf{h}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D) are statistically independent ie. probability of ![h_i](http://latex.codecogs.com/gif.latex?h_i) is independent of ![h_j](http://latex.codecogs.com/gif.latex?h_j) where ![i\neq j](http://latex.codecogs.com/gif.latex?i%5Cneq%20j).

`page 480` One way of looking at linear factor model is too see that we are basically assuming that the data distribution can be approximated by a linear transformation of a latent distribution. We account for the error in the approximation with the noise term. So if we use a Gaussian as the latent distribution, we assume that the data distribution can be approximated by a linearly transformed Gaussian.

`page 483` To recall, Equation 3.47 is 

![p_x(\mathbf{x})=p_y(g(\mathbf{x}))\left|\text{det}\left(\frac{\partial g(\mathbf{x})}{\partial\mathbf{x}}\right)\right|](http://latex.codecogs.com/gif.latex?p_x%28%5Cmathbf%7Bx%7D%29%3Dp_y%28g%28%5Cmathbf%7Bx%7D%29%29%5Cleft%7C%5Ctext%7Bdet%7D%5Cleft%28%5Cfrac%7B%5Cpartial%20g%28%5Cmathbf%7Bx%7D%29%7D%7B%5Cpartial%5Cmathbf%7Bx%7D%7D%5Cright%29%5Cright%7C)

where ![\mathbf{y}=g(\mathbf{x})](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D%3Dg%28%5Cmathbf%7Bx%7D%29).

Or in scalar form, Equation 3.46

![p_x(x)=p_y(g(x))\left|\frac{\partial g(x)}{\partial x}\right|](http://latex.codecogs.com/gif.latex?p_x%28x%29%3Dp_y%28g%28x%29%29%5Cleft%7C%5Cfrac%7B%5Cpartial%20g%28x%29%7D%7B%5Cpartial%20x%7D%5Cright%7C)

where ![y=g(x)](http://latex.codecogs.com/gif.latex?y%3Dg%28x%29).

`page 487` Here's a walkthrough for deriving Equation 13.18.

We begin with Equation 13.17: ![\text{arg max }\log p(\mathbf{h}\mid\mathbf{x})](http://latex.codecogs.com/gif.latex?%5Ctext%7Barg%20max%20%7D%5Clog%20p%28%5Cmathbf%7Bh%7D%5Cmid%5Cmathbf%7Bx%7D%29).

Using Bayes' Theorem, we have 

![\text{arg max }\log\left(\frac{p(\mathbf{x}\mid\mathbf{h})p(\mathbf{h})}{p(\mathbf{x})}\right)=\text{arg max }(\log p(\mathbf{x}\mid\mathbf{h})+\log p(\mathbf{h})-\log p(\mathbf{x}))](http://latex.codecogs.com/gif.latex?%5Ctext%7Barg%20max%20%7D%5Clog%5Cleft%28%5Cfrac%7Bp%28%5Cmathbf%7Bx%7D%5Cmid%5Cmathbf%7Bh%7D%29p%28%5Cmathbf%7Bh%7D%29%7D%7Bp%28%5Cmathbf%7Bx%7D%29%7D%5Cright%29%3D%5Ctext%7Barg%20max%20%7D%28%5Clog%20p%28%5Cmathbf%7Bx%7D%5Cmid%5Cmathbf%7Bh%7D%29&plus;%5Clog%20p%28%5Cmathbf%7Bh%7D%29-%5Clog%20p%28%5Cmathbf%7Bx%7D%29%29).

Then we substitute in Equation 13.13 and drop the last term ![\log p(\mathbf{x})](http://latex.codecogs.com/gif.latex?%5Clog%20p%28%5Cmathbf%7Bx%7D%29) (since it doesn't depend on ![\mathbf{h}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D)), to get 

![\text{arg max }(\log p(\mathbf{x}\mid\mathbf{h})+\log \frac{\lambda}{4}e^{-\frac{1}{2}\lambda||\mathbf{h}||_1})](http://latex.codecogs.com/gif.latex?%5Ctext%7Barg%20max%20%7D%28%5Clog%20p%28%5Cmathbf%7Bx%7D%5Cmid%5Cmathbf%7Bh%7D%29&plus;%5Clog%20%5Cfrac%7B%5Clambda%7D%7B4%7De%5E%7B-%5Cfrac%7B1%7D%7B2%7D%5Clambda%7C%7C%5Cmathbf%7Bh%7D%7C%7C_1%7D%29)


![=\text{arg max }(\log p(\mathbf{x}\mid\mathbf{h})+\log \frac{\lambda}{4}-\frac{1}{2}\lambda||\mathbf{h}||_1)](http://latex.codecogs.com/gif.latex?%3D%5Ctext%7Barg%20max%20%7D%28%5Clog%20p%28%5Cmathbf%7Bx%7D%5Cmid%5Cmathbf%7Bh%7D%29&plus;%5Clog%20%5Cfrac%7B%5Clambda%7D%7B4%7D-%5Cfrac%7B1%7D%7B2%7D%5Clambda%7C%7C%5Cmathbf%7Bh%7D%7C%7C_1%29)

Then we can substitute Equation 13.12 (Gaussian PDF) and drop the ![\log \frac{\lambda}{4}](http://latex.codecogs.com/gif.latex?%5Clog%20%5Cfrac%7B%5Clambda%7D%7B4%7D) term (again it doesn't depend on ![\mathbf{h}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D)), to get

![\text{arg max }(\log\sqrt{\frac{\beta}{2\pi}}e^{-\frac{\beta(\mathbf{x}-\mathbf{Wh}-\mathbf{b})^2}{2}}-\frac{1}{2}\lambda||\mathbf{h}||_1)](http://latex.codecogs.com/gif.latex?%5Ctext%7Barg%20max%20%7D%28%5Clog%5Csqrt%7B%5Cfrac%7B%5Cbeta%7D%7B2%5Cpi%7D%7De%5E%7B-%5Cfrac%7B%5Cbeta%28%5Cmathbf%7Bx%7D-%5Cmathbf%7BWh%7D-%5Cmathbf%7Bb%7D%29%5E2%7D%7B2%7D%7D-%5Cfrac%7B1%7D%7B2%7D%5Clambda%7C%7C%5Cmathbf%7Bh%7D%7C%7C_1%29)

![=\text{arg max }(\log\sqrt{\frac{\beta}{2\pi}}{-\frac{\beta(\mathbf{x}-\mathbf{Wh}-\mathbf{b})^2}{2}}-\frac{1}{2}\lambda||\mathbf{h}||_1)](http://latex.codecogs.com/gif.latex?%3D%5Ctext%7Barg%20max%20%7D%28%5Clog%5Csqrt%7B%5Cfrac%7B%5Cbeta%7D%7B2%5Cpi%7D%7D%7B-%5Cfrac%7B%5Cbeta%28%5Cmathbf%7Bx%7D-%5Cmathbf%7BWh%7D-%5Cmathbf%7Bb%7D%29%5E2%7D%7B2%7D%7D-%5Cfrac%7B1%7D%7B2%7D%5Clambda%7C%7C%5Cmathbf%7Bh%7D%7C%7C_1%29)

Finally we can drop all constants and terms not depending on ![\mathbf{h}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D) (and assuming no bias term), to get 

![\text{arg max }(-\beta||\mathbf{x}-\mathbf{Wh}||_2^2-\lambda||\mathbf{h}||_1)=\text{arg min }(\beta||\mathbf{x}-\mathbf{Wh}||_2^2+\lambda||\mathbf{h}||_1)](http://latex.codecogs.com/gif.latex?%5Ctext%7Barg%20max%20%7D%28-%5Cbeta%7C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7BWh%7D%7C%7C_2%5E2-%5Clambda%7C%7C%5Cmathbf%7Bh%7D%7C%7C_1%29%3D%5Ctext%7Barg%20min%20%7D%28%5Cbeta%7C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7BWh%7D%7C%7C_2%5E2&plus;%5Clambda%7C%7C%5Cmathbf%7Bh%7D%7C%7C_1%29)

`page 488` [Here](http://www.suhasmathur.com/the-bayesian-observer/2017/1/7/spike-and-slab-priors) is a blog post describing the spike and slab prior.

`page 489` Again, the factorial prior refers to the assumption that the elements of the individual features are independent of each other.
