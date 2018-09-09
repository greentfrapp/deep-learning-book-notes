# 19 Approximate Inference

`page 626` I believe that in the **Expectation Maximization (EM)** algorithm, the equation for the E-step should probably be

![q(\mathbf{h}^{(i)}\mid\mathbf{v})=p(\mathbf{h}^{(i)}\mid\mathbf{v}^{(i)};\mathbf{\theta}^{(t-1)})](http://latex.codecogs.com/gif.latex?q%28%5Cmathbf%7Bh%7D%5E%7B%28i%29%7D%5Cmid%5Cmathbf%7Bv%7D%29%3Dp%28%5Cmathbf%7Bh%7D%5E%7B%28i%29%7D%5Cmid%5Cmathbf%7Bv%7D%5E%7B%28i%29%7D%3B%5Cmathbf%7B%5Ctheta%7D%5E%7B%28t-1%29%7D%29)

rather than

![q(\mathbf{h}^{(i)}\mid\mathbf{v})=p(\mathbf{h}^{(i)}\mid\mathbf{v}^{(i)};\mathbf{\theta}^{(0)})](http://latex.codecogs.com/gif.latex?q%28%5Cmathbf%7Bh%7D%5E%7B%28i%29%7D%5Cmid%5Cmathbf%7Bv%7D%29%3Dp%28%5Cmathbf%7Bh%7D%5E%7B%28i%29%7D%5Cmid%5Cmathbf%7Bv%7D%5E%7B%28i%29%7D%3B%5Cmathbf%7B%5Ctheta%7D%5E%7B%280%29%7D%29)

In other words, we set ![q(\mathbf{h}^{(i)}\mid\mathbf{v})](http://latex.codecogs.com/gif.latex?q%28%5Cmathbf%7Bh%7D%5E%7B%28i%29%7D%5Cmid%5Cmathbf%7Bv%7D%29) to the updated ![\mathbf{\theta}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Ctheta%7D) from the previous M-step.

`page 628`

> We can thus justify a learning procedure similar to EM, in which we alternate between performing MAP inference to infer ![\mathbf{h}^\ast](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bh%7D%5E%5Cast) and then update ![\mathbf{\theta}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Ctheta%7D) to increase ![\log p(\mathbf{h}^\ast,\mathbf{v})](http://latex.codecogs.com/gif.latex?%5Clog%20p%28%5Cmathbf%7Bh%7D%5E%5Cast%2C%5Cmathbf%7Bv%7D%29).

This is actually similar to the way we perform supervised deep learning. Given each (mini)batch of training samples, we perform forward propagation and calculate the intermediate hidden/latent variables, in order to calculate the loss. Then we update the model parameters by performing gradient descent on the loss.
