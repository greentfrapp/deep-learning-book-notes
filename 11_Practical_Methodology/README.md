# 11 Practical Methodology

`page 409`

> Many of the recommendations in this chapter are adapted from Ng (2015).

[This](https://see.stanford.edu/materials/aimlcs229/ML-advice.pdf) is the link to Andrew Ng's cited slides.

Andrew Ng is a great teacher with several great online educational resources available (he also founded Coursera, based on his initial online machine learning course). Another great resource for practical methodology is Ng's book (work-in-progress as of writing) Machine Learning Yearning, which can be found [here](https://gallery.mailchimp.com/dc3a7ef4d750c0abfc19202a3/files/d2dee348-4ada-400c-a0b2-d884fcdc368f/Ng_MLY01_11.pdf)

`page 411`

> One way to solve this problem is to instead measure precision and recall.

Mathematically, **precision** is defined as ![\frac{\text{TruePositives}}{\text{TruePositives}+\text{FalsePositives}}](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Ctext%7BTruePositives%7D%7D%7B%5Ctext%7BTruePositives%7D&plus;%5Ctext%7BFalsePositives%7D%7D), while **recall** is defined as ![\frac{\text{TruePositives}}{\text{TruePositives}+\text{FalseNegatives}}](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Ctext%7BTruePositives%7D%7D%7B%5Ctext%7BTruePositives%7D&plus;%5Ctext%7BFalseNegatives%7D%7D).

(It can get a bit confusing to remember whether precision or recall incorporates false positives or false negatives. I like to remember the equations by thinking of how 'precision' begins with **P** so it incorpoates false **p**ositives.)

A concrete way of understanding precision and recall is with disease prediction. Suppose there is a disease detector that has 0.9 precision. This means that if the detector says you have the disease, you have a 90% chance of actually having the disease. On the other hand, if the detector has 0.9 recall, this means that if you have the disease, the detector has a 90% chance of detecting it.

As mentioned in the text, if the detector says no one has the disease, the detector has perfect precision and zero recall. Perfect precision because it will never give a positive prediction so obviously it will never give a wrong positive prediction. Zero recall because if you have the disease, the detector has a 0% chance of detecting it.

Also mentioned in the text, if the detector says everyone has the disease, the detector has precision equal to the percentage of people who have the disease and perfect recall. Precision equal to the percentage of people who have the disease because your probability of having the disease, conditioned on the detector detecting the disease, is the same as the prior probability of having the disease. It has perfect recall because if you have the disease the detector has a 100% chance of detecting it.

Notice that using precision and recall as metrics (and subsequently the F1), is much better than accuracy when we deal with imbalanced datasets, where the frequency of one class might be several magnitudes smaller than another class.

`page 412` In order to consider coverage, the machine learning system has to be able to calculate its confidence in its prediction. In general, we might try using the likelihood scores predicted by the system as a measure of confidence, such as the sigmoid (for binary classification) or softmax (for multiclass classification) scores. Alternatively, we can have another neural network that predicts the confidence of the prediction. 

A threshold is then used to determine when a human should intervene. This threshold is tuned with the coverage metric, as well as the resulting F1 when the system makes the prediction.

`page 413` Slightly unrelated, but research-wise, a common lowerbound baseline used when introducing new tasks and datasets is simply the random baseline, which is the score of a completely random model. However, there are some cases where a purely random baseline should not be the lowerbound. For example, in Agrawal et al.'s work [2015](https://arxiv.org/abs/1505.00468v6) introducing the Visual Question Answering dataset, they used a 'prior ("yes")' baseline, which basically answered 'yes' to all the questions and has higher-than-random accuracy due to the distribution of the questions (See Table 2 in original paper).

`page 413` As implied by the text, initial baselines can simply be vanilla traditional algorithms that have been proven to work generally well and can be implemented quickly to get an initial result.

But for those looking for the latest state-of-the-art, [here](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems) is a GitHub repo that contains links to implemented state-of-the-art algorithms across different domains and tasks.

`page 417`

> If you have time to tune only one hyperparameter, tune the learning rate.

Interesting piece of advice! Not sure if I agree or disagree, mainly because I found that my go-to optimizer Adam generally works well with a learning rate of 1e-3 or 1e-4, although to be honest I haven't spent a lot of time purposefully tuning the learning rate.

`page 418`

> Your goal is to reduce this gap without increasing training error faster than the gap decreases.

This effectively means that your test error (training error + generalization error/gap) decreases.

`page 419` A nice summary of a few hyperparameters. It is a good exercise to try to come up with a similar table for more hyperparameters such as batch size, number of training steps (early stopping), weights and bias initializer etc., as well as other algorithm-specific hyperparameters.

`page 420`

> [...] we are trying to find a value of the hyperparameters that optimizes an objective function, such as validation error [...]

It is important to see and understand that we are optimizing the validation error here and not the training or test error.

`page 422` It is counter-intuitive and definitely interesting that random search appears to work better than grid search for hyperparameter tuning. A short explanation is given in the text and detailed by Bergstra et al. ([2011](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)).

`page 424` More recently, neural architecture search (NAS) has shown some promise in the domain of hyperparameter optimization. 

Suppose we are designing an image classifier. The paradigm adopted by NAS is to let another machine learning algorithm learn the hyperparameters for the image classifier automatically, including the entire architecture of the classifier, such as how the convolutional layers connect to each other, where to place pooling layers etc. 

NASNet by Zoph et al. ([2017](https://arxiv.org/abs/1707.07012)) is one of the more well-known works on NAS, applying the technique to design an architecture for ImageNet. To quote Zoph et al., NASNet "is 1.2% better in top-1 accuracy than the best human-invented architectures while having 9 billion fewer FLOPS - a reduction of 28% in computational demand from the previous state-of-the-art model".

`page 424` **Important Section!** I urge anyone reading this to pay a ton of attention to this section on **Debugging Strategies**.

From personal experience, many novice researchers and practitioners (including myself!) tend to put immediate blame on poor hyperparameters or bad data when an implementation does not perform as expected. This results in many hours, days or weeks wasted tuning hyperparameters or collecting/cleaning data when the problem might be a bug in the implementation.

I definitely recommend doing some form of unit testing to make sure each part of the implementation works as intended. This section also recommends several other very good practices that I had to learn myself from a lot of wasted effort. So read this section closely and use the advice!
