# 2 Linear Algebra

`page 30` Just a note-to-self: indices for a 2D matrix refers to row then column eg. 

![A_{1,2}](http://latex.codecogs.com/gif.latex?A_{1,2})

refers to element in row 1 and column 2 of matrix A.

`page 32` Broadcasting is more than just a notation convenience. When using libraries such as numpy and Tensorflow, broadcasting can reduce the memory requirements of a program, compared to alternatives such as `tf.tile`.

`page 33` When using neural networks for regression, a way of looking at the problem is to see the final layer of the network as a system of linear equations, assuming no activation.

![\mathbf{y}_{pred}=\mathbf{Wf}+\mathbf{b}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D_%7Bpred%7D%3D%5Cmathbf%7BWf%7D&plus;%5Cmathbf%7Bb%7D)

where ![\mathbf{y}_{pred}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7By%7D_%7Bpred%7D) is the predicted output, ![\mathbf{f}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bf%7D) is the feature vector output from the second last layer and ![\mathbf{W}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BW%7D) and ![\mathbf{b}](http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bb%7D) are the weights and bias of the final layer. Assuming we fix the earlier layers, we can then solve directly for the weights and bias given enough samples, although such a solution is not guaranteed to generalize to unseen samples.

`page 37` Norms

`page 38` Cosine similarity

`page 41` eigendecomposition observations - matrix is singular if and only if any eigenvalues are zero, since this implies that a linear combination of the columns give zero, which means that the columns are linearly dependent and the span of the matrix is less than R^n. Show also how non-eigenvectors change after multiplication wrt Figure 2.3. Also quadratic expression propertiees in paragraph 1 of page 42. 

`page 45` Interpretation of determinant in context of eigenvalue scaling

`page 49` Proof by induction for PCA.

