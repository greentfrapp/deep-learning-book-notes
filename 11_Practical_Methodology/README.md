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

`page 413` [Here](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems) is a GitHub repo that contains links to state-of-the-art algorithms across different domains and tasks.

