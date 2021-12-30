# **Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization - Week2**

## **Optimization algorithms**

These techniques will allow to run the DL models faster.

### **Mini-batch Gradient Descent**

Vectorization allows to make calculations on m examples, which are processed in batch. However, if m is quite big, it is unfeasible to process all the training instances at once. So, we can divide the training set into smaller partitions called mini-batches (X and Y matrices), and calculate gradient descent algorithm on each mini-batch.

The progress of this algorithm on the cost function may not decrease on every iteration because in each step the training set is different. So, this plot would like noisier than the batch gradient descent. We need to choose the size of the mini-batches. If mini-batch size is m, this is the same as batch gradient descend, which is guaranteed to converge. But, if mini-batch size is 1 (each example is its own mini-batch) this algorithm is called stochastic gradient descent and it won't ever converge, it will always oscillate around the region of the minimum. In practice, a good choice for the mini-batch size is a value in between both extremes, which provides the fastest learning. There are some guidelines to choose the specific batch-size value:

* Small datasets (<2000) - batch gradient descent
* Big datasets - mini-batch size should be a power of two (64, 128, 256, 1024, etc), and the mini-batches should fit in CPU/GPU memory. This value can be also be an hyperparameter to be tuned.

### **Exponentially Weighted Averages**

This is a key component of several optimization algorithms. The following formula is used to calculate this component:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large V_{\t} = \beta V_{t-1} %2B \left(1-\beta\right)\theta_{t}"/>
</p>

The beta parameter is an hyperparameter to obtain the best results with this algorithm. All the Vt values create an exponentially decaying function, and the final output is a single number with the sums of the previous calculations.

To make the calculations of exponentially weighted averages more accurate, we need to apply the bias correction, which helps to avoid very low values at the first estimations and obtain accurate values early on.

### **Gradient Descent with Momentum**

The basic idea of this algorithm is to calculate the exponentially weighted average of gradients, an then use these values to update the weights.

In the implementation of this algorithm, we update the weights with the weighted average of derivatives. In this way, the algorithm takes steps that are smaller oscillations in the vertical direction, but these movements are more directed in the horizontal direction, which allows to take a straightforward path to the minimum.

THere are two hyperparameters, the learning rate alpha and beta that controls the exponentially weighted average.

### **RMSprop**

This algorithm also applies the exponentially weighted average, but it updates the weights subtracting the learning rate times the derivatives divided bt the root square of the exponentially weighted values.

### **Adaptive moment estimation (Adam) Optimization Algorithm**

This algorithm merges both gradient descent with momentum and RMSprop in a single method. In this way, we have two betas as hyperparameters of the algorithm, apply bias correction to both betas, and perform the update of weights with both betas.

There are three hyperparameters, the alpha learning rate, the two betas, and epsilon parameter for bias correction.

### **Learning rate decay**

This method consists on slowly reduce the learning rate over time, and it helps the optimization algorithm to converge. So, the steps are small on the initial stages, but they decrease in the late steps.

This algorithm adds the hyperparameter decay rate to the learning process. There are different implementation of the learning rate decay such as exponentially decay, manual decay, among others.

### **The Problem of Local Optima**

The intuition of local minimums in low dimensional spaces are not translated to high dimensional spaces, in which there are are saddle points instead. However, plateaus can slow down the learning process, which are regions where the derivatives are close to zero for a long time.

So, it is unlikely to get stuck in a bad local optima, but plateaus can make learning process slow.  
