# **Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization - Week1**

## **Setting up your Machine Learning Application**

### **Train/Dev/Test sets**

The correct splitting of train, validation/dev, and test datasets can help to have a good performance on the neural networks (NN).

When you are training a NN, there are some hyperparameters to choose, including the number of layers, the number of hidden units, learning rates, activation functions, among others. It is unlikely to choose the best hyperparameters at the beginning, so we need to set them on the NNs, train the models, evaluate the models, and maintain or change the hyperparameters. So, now deep learning is an iterative process - Idea/Code/Experiment.

Usually, the entire dataset is divided into three parts:

1. Training set: data used to train models.
2. Hold-out cross validation/development set: data utilized for hyperparameter-tuning, or to decide the best models.
3. Test set: data to evaluate the best model to get an unbiased estimate of how well the algorithm is doing.

In machine learning models, it was common to create the partitions with 70/30 or 60/20/20 proportions. These ratios are a good rule of thumb if your dataset is small (until 10K instances). However, if you are working with huge amounts of data (around millions), the trend is that test and dev datasets to be smaller percentages of the entire dataset (around 1%, or less depending on the size of the dataset).

Also, it is important to use mismatched train/test distributions. This means that both datasets have similar instances in terms of quality, size, and other features.

Finally, with deep learning projects it is ok not having a test set, so we only work with training and dev datasets. There are some misunderstandings with the terminology when there are only training and test/dev datasets.

### **Bias/Variance**

The train and dev datasets errors can be used to evaluate potential bias and variance issues of ML models. In general, the performance of a DL model can have three possible scenarios:

* Underfitting (high bias - high train and dev errors): the model can't make the predictions properly because it is too simple.
* Overfitting: (high variance - low train error and high dev error): the model performs well on the training data, but it can't generalize the predictions to external data.
* Just right (low bias and variance - low train and dev errors): an equilibrium of the two previous scenarios.
* Worst case (have high variance and bias - high bias and variance, high train error and higher dev error).

Also, we need to consider the optimal or Bayes error, which can affect the perception of high/low values of train and dev errors.

### **Basic Recipe for Machine Learning**

In general, after training an initial model, we should follow these steps:

1. Reviewing if the model has high bias looking at the training dataset performance. To avoid this issue, we can train a bigger network, train during longer periods of time, or try other NN architectures.
2. Reviewing if the model has high variance looking at the dev dataset performance. To avoid this issue, we can train models with more data, apply regularization techniques, or try other NN architectures.

In the earlier era of ML, there was an agreement in the existence of the bias-variance tradeoff, which tells that you can not increase bias or variance without decreasing the other aspect. However, if you have a big dataset and train DL models, you can have decrease bias or variance without affecting the other parameter.

## **Regularizing your Neural Network**

### **Regularization**

This technique is used to improve models with hig variance, or that are overfitting. In logistic regression, regularizing a model means the addition of a regularization term with respect to the w parameter in the cost function. Usually, the regularization of the b parameter is omitted because it is just a number.

In general, there are two types of regularization, L1 and L2, which differs in the norm or scaling factor they use. The most common regularization is L2, which applies the euclidian norm to improve the model. Also, there is a lambda regularization parameter, which is a hyperparameter for tuning on the dev dataset.

In neural networks, the idea of regularization is similar to logistic regression, but it is applied to the w matrix of parameters. The norm of the matrix is known as *Frobenius norm* for linear algebra conventions. L2 regularization is also known as weight decay.

### **Why Regularization Reduces Overfitting?**

In NNs, the regularization is the addition of a term that penalizes the weight matrices from being too large, which corresponds to the Frobenius norm. The regularization term will cause that some weights of the parameter matrix would be close to zero, and some hidden units will have a small effect on the model. So, the model will end up being simpler than before, and it will be less prone to overfitting. Also, when the regularization is applied, activation functions would be linear, and it make the model less able to overfit.

### **Dropout regularization**

With this regularization, we go through each of the layers of the network and set some probability of eliminating a node in the NN. So, we use a different version of the NN for each training example. The regularization occur because we train smaller NN each time.

The most common technique to implement dropout is by the inverted dropout method. First, we need to create random numbers less than a keep-probability value, which defines the probability that a hidden unit will be kept in the NN. The essential part of this method is the division by the keep-probability. During test time, the dropout is not applied because we don't want the outputs to be random.

### **Understanding Dropout**

With dropout, the inputs can get randomly eliminated, so the hidden units can't rely on any one feature because all of them could go away at random. Thus, we have to spread out the weights, which will shrink the squared norm of the weights. In this way dropout can be shown to be an adaptive form of L2 regularization, and the effect of both methods are similar. One of the main downsides of dropout is that the cost function is not well defined anymore, and it's more difficult to calculate.

### **Other Regularization Methods**

#### **Data augmentation**

This technique take the training dataset and augment them by adding some small changes such as rotations, distortions, amon others.

#### **Early stopping**

This method stops the training process at the point in which development error starts to increase. In this way, the NN will be smaller, and this can help to regularize the model. The main downside of early stopping is that it is not possible to work independently on both tasks of optimize the cost function and not overfitting.

If you have enough computational power, it is recommended to use L2 regularization with different values of lambda.

## **Setting Up your Optimization Problem**

### **Normalizing inputs**

In this process, there are two steps: subtract mean and normalize the variance. The same normalization method (mu and sigma values) should be applied to training and test datasets.

Normalizing helps to the convergence of the cost function because if features have similar scales, the cost function will be more round and easier to optimize. Thus, the training of the algorithm runs faster with normalized features.

### **Vanishing/Exploding gradients**

When you train deep networks, the derivatives or slopes can get either very big or very small, which makes training difficult.

The process is caused because the activation or gradients increase or decrease exponentially as a function of the number of layers. In this way the gradient descent algorithm will take a long time to learn something.

### **Weight initialization for Deep Networks**

One alternative to solve the problem of vanishing/exploding gradients is a careful choice of the random initialization for the NN. With this process, the weight matrices values would not be bigger than 1 and not too much less than 1. In this way, gradients don't explode or vanish too quickly.

Usually, we use the RELU activation function, but we can also use the TANH (Xavier initialization), and other functions. But in practice, these formulas are a starting point or default values to use for the variance of the initialization for the weight matrices. These parameters can be tunned as hyperparameters during the learning process.

### **Numerical approximation of gradients**

This is a technique required to make sure if the implementation of back propagation is correct. This method consists of taking a two sided difference, which allows to numerically verify wether or not a function is a correct implementation of the derivative of a function.

### **Gradient checking**

This method helps to fund bugs in the implementation of back propagation. The first step is to take the wight matrices and reshape them into a big vector theta, which is also applied for the derivative matrices. So, the question is if d-theta vector is the gradient of J(theta).

To check if the back propagation is correct, we need to apply the formula of gradient checking and verify its results.

This process should not be applied in training, only on the debug step. Also, we need to consider the regularization if it was applied, and that gradient checking does not work with dropout.

### **Initialization programming assignment**

* Different initializations lead to very different results
* Random initialization is used to break symmetry and make sure different hidden units can learn different things
* Resist initializing to values that are too large!
* He initialization works well for networks with ReLU activations
