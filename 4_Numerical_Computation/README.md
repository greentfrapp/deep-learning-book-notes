# 4 Numerical Computation

`page 78` The text describes a simple way of evaluating ![\text{softmax}(\mathbf{x})_i=\frac{\text{exp}(x_i)}{\sum^n_{j=1}\text{exp}(x_j)}](http://latex.codecogs.com/gif.latex?%5Ctext%7Bsoftmax%7D%28%5Cmathbf%7Bx%7D%29_i%3D%5Cfrac%7B%5Ctext%7Bexp%7D%28x_i%29%7D%7B%5Csum%5En_%7Bj%3D1%7D%5Ctext%7Bexp%7D%28x_j%29%7D) by seeing that ![\text{softmax}(\mathbf{x})=\text{softmax}(\mathbf{z})](http://latex.codecogs.com/gif.latex?%5Ctext%7Bsoftmax%7D%28%5Cmathbf%7Bx%7D%29%3D%5Ctext%7Bsoftmax%7D%28%5Cmathbf%7Bz%7D%29), where ![\mathbf{z}=\mathbf{x}-\text{max}_ix_i](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bz%7D%3D%5Cmathbf%7Bx%7D-%5Ctext%7Bmax%7D_ix_i). Evaluating ![\text{softmax}(\mathbf{z})](http://latex.codecogs.com/gif.latex?%5Ctext%7Bsoftmax%7D%28%5Cmathbf%7Bz%7D%29) instead then rules out the possibility of overflow and underflow.

The proceeding paragraph then proposes the problem of implementing ![\text{log softmax}(\mathbf{x})](http://latex.codecogs.com/gif.latex?%5Ctext%7Blog%20softmax%7D%28%5Cmathbf%7Bx%7D%29), the problem being that if any element of x is very negative, we get a softmax value of 0 and will have to evaluate log(0), which causes underflow. A numerically stable way of implementing the function is left to the reader as an exercise, so here's a possible solution.

Since ![\text{softmax}(\mathbf{x})=\text{softmax}(\mathbf{z})](http://latex.codecogs.com/gif.latex?%5Ctext%7Bsoftmax%7D%28%5Cmathbf%7Bx%7D%29%3D%5Ctext%7Bsoftmax%7D%28%5Cmathbf%7Bz%7D%29), where ![\mathbf{z}=\mathbf{x}-\text{max}_ix_i](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bz%7D%3D%5Cmathbf%7Bx%7D-%5Ctext%7Bmax%7D_ix_i), then 

![\text{log softmax}(\mathbf{x})_i=\text{log softmax}(\mathbf{z})_i](http://latex.codecogs.com/gif.latex?%5Ctext%7Blog%20softmax%7D%28%5Cmathbf%7Bx%7D%29_i%3D%5Ctext%7Blog%20softmax%7D%28%5Cmathbf%7Bz%7D%29_i)

![\text{log softmax}(\mathbf{x})_i=\text{log}(\frac{\text{exp}(z_i)}{\sum^n_{j=1}\text{exp}(z_j)})](http://latex.codecogs.com/gif.latex?%5Ctext%7Blog%20softmax%7D%28%5Cmathbf%7Bx%7D%29_i%3D%5Ctext%7Blog%7D%28%5Cfrac%7B%5Ctext%7Bexp%7D%28z_i%29%7D%7B%5Csum%5En_%7Bj%3D1%7D%5Ctext%7Bexp%7D%28z_j%29%7D%29)

![\text{log softmax}(\mathbf{x})_i=\text{log}(\text{exp}(z_i))-\text{log}\sum^n_{j=1}\text{exp}(z_j)](http://latex.codecogs.com/gif.latex?%5Ctext%7Blog%20softmax%7D%28%5Cmathbf%7Bx%7D%29_i%3D%5Ctext%7Blog%7D%28%5Ctext%7Bexp%7D%28z_i%29%29-%5Ctext%7Blog%7D%5Csum%5En_%7Bj%3D1%7D%5Ctext%7Bexp%7D%28z_j%29)

![\text{log softmax}(\mathbf{x})_i=z_i-\text{log}\sum^n_{j=1}\text{exp}(z_j)](http://latex.codecogs.com/gif.latex?%5Ctext%7Blog%20softmax%7D%28%5Cmathbf%7Bx%7D%29_i%3Dz_i-%5Ctext%7Blog%7D%5Csum%5En_%7Bj%3D1%7D%5Ctext%7Bexp%7D%28z_j%29)

Like the softmax example from before, here the maximum value of ![z_j](http://latex.codecogs.com/gif.latex?z_j) is 0, which prevents overflow from evaluating the exponential. And at least one term in the summation has a value of 1, which prevents underflow from evaluating log(0).

`page 79`

> Optimization refers to the task of either minimizing or maximizing some function f(x) by altering x.

In the more specific notation and context of common machine learning algorithms, we typically minimize ![f(\mathbf{x},\mathbf{y},\mathbf{\theta})](http://latex.codecogs.com/gif.latex?f%28%5Cmathbf%7Bx%7D%2C%5Cmathbf%7By%7D%2C%5Cmathbf%7B%5Ctheta%7D%29) by altering ![\theta](http://latex.codecogs.com/gif.latex?%5Ctheta), which refers to the parameters of the neural network, while ![\mathbf{x}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D) and ![\mathbf{y}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D) refer to the training samples and training labels respectively.

`page 82` Another commonly-shown equation to illustrate gradient descent is:

![\theta\leftarrow\theta-\alpha\frac{\partial L}{\partial\theta}](http://latex.codecogs.com/gif.latex?%5Ctheta%5Cleftarrow%5Ctheta-%5Calpha%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%5Ctheta%7D)

where ![\theta](http://latex.codecogs.com/gif.latex?%5Ctheta) refers to the neural network parameters, ![\alpha](http://latex.codecogs.com/gif.latex?%5Calpha) refers to the learning rate, ![L](http://latex.codecogs.com/gif.latex?L) refers to the value of the loss function and the left arrow is an assignment operator. We move ![\theta](http://latex.codecogs.com/gif.latex?%5Ctheta) in the opposite direction of the gradient in order to reduce the loss.

`page 88` If a function is Lipschitz continuous, the gradient/slope at any point is less than or equal to a constant value (denoted as the Lipschitz constant). Note that this is not a universal constant and is function-specific. For instance, ![f(x)=2x](http://latex.codecogs.com/gif.latex?f%28x%29%3D2x) has a Lipschitz constant of 2 since its derivative has a constant value of 2. Another example, ![f(x)=\sin(x)](http://latex.codecogs.com/gif.latex?f%28x%29%3D%5Csin%28x%29) has a Lipschitz constant of 1 since its derivative is ![\cos(x)](http://latex.codecogs.com/gif.latex?%5Ccos%28x%29), which has a maximum magnitude of 1.

There is actually a cool illustration of a [Lipschitz continuous](https://en.wikipedia.org/wiki/Lipschitz_continuity) function that can be found on Wikipedia (see below).

![Illustration of a Lipschitz continuous function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Lipschitz_continuity.png/220px-Lipschitz_continuity.png)

With reference to the above illustration, the Lipschitz constant gives the magnitude of the gradients/slopes of the double cone. Since the gradient/slope of the function never exceeds this value, we can move vertex/middle of the double cone along the function and the function will always remain outside of the double cone (in white).

`page 92` ToDo: Implement example with numpy