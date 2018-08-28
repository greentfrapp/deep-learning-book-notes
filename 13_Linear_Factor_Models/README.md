# 13 Linear Factor Models

`page 480` Here, a factorial distribution means that the components of ![\mathbf{h}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D) are statistically independent ie. probability of ![h_i](http://latex.codecogs.com/gif.latex?h_i) is independent of ![h_j](http://latex.codecogs.com/gif.latex?h_j) where ![i\neq j](http://latex.codecogs.com/gif.latex?i%5Cneq%20j).

`page 480` One way of looking at linear factor model is too see that we are basically assuming that the data distribution can be approximated by a linear transformation of a latent distribution. We account for the error in the approximation with the noise term. So if we use a Gaussian as the latent distribution, we assume that the data distribution can be approximated by a linearly transformed Gaussian.

`page 483` To recall, Equation 3.47 is 

![p_x(\mathbf{x})=p_y(g(\mathbf{x}))\left|\text{det}\left(\frac{\partial g(\mathbf{x})}{\partial\mathbf{x}}\right)\right|](http://latex.codecogs.com/gif.latex?p_x%28%5Cmathbf%7Bx%7D%29%3Dp_y%28g%28%5Cmathbf%7Bx%7D%29%29%5Cleft%7C%5Ctext%7Bdet%7D%5Cleft%28%5Cfrac%7B%5Cpartial%20g%28%5Cmathbf%7Bx%7D%29%7D%7B%5Cpartial%5Cmathbf%7Bx%7D%7D%5Cright%29%5Cright%7C)

where ![\mathbf{y}=g(\mathbf{x})](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D%3Dg%28%5Cmathbf%7Bx%7D%29).

Or in scalar form, Equation 3.46

![p_x(x)=p_y(g(x))\left|\frac{\partial g(x)}{\partial x}\right|](http://latex.codecogs.com/gif.latex?p_x%28x%29%3Dp_y%28g%28x%29%29%5Cleft%7C%5Cfrac%7B%5Cpartial%20g%28x%29%7D%7B%5Cpartial%20x%7D%5Cright%7C)

 where ![y=g(x)](http://latex.codecogs.com/gif.latex?y%3Dg%28x%29).
 