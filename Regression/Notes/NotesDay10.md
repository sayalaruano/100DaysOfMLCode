# **NotesDay10**

## **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow chapter 4 - Training Models**

Knowing the implementations details of ML models is not always required. But, acquiring this knowledge could be helpful to choose the proper 
model, training algorithm, and the set of hyperparameters for a particular problem. Also, it can contribute to debug failures and perform error   
analysis efficiently. 

### **Linear regression (LR)**

The LR is a linear model that makes a prediction by calculating a **weighted sum of the input features** and the **bias term**. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \hat{y} = \Theta_{0} %2B\Theta_{1}x_{1} %2B\Theta_{2}x_{2} %2B ... %2B\Theta_{n}x_{n}"/>
</p>

This formula has a 
vectorized form, computing the dot product between parameter (bias term and feature weights) and feature (X0 is always equal to 1) vectors. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \hat{y} = h_{\Theta}\left( x \right) = \Theta \cdot x"/>
</p>

Training a model means setting its combination of parameters so that the model best fits the training set, and minimizes the cost function. The RMSE is a measure of how well the model fits the 
training data, so the aim is finding a parameter vector that minimizes the RMSE. In fact, it is convenient to use the MSE instead of RMSE because 
the first one is easier to minimize and both produce the same result. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large MSE\left(X,h_{\Theta}\right) = \frac{1}{m} \sum_{i=1}^{m}\left(\Theta^{T}x^{\left(i\right)}-y^{\left(i\right)}\right)^{2}"/>
</p>

#### **Normal Equation** 

Mathematical equation that calculates the parameter vector of the LR model. This is a closed-form solution. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \hat{\Theta}=\left(X^{T}  X\right)^{-1}X^{T}y"/>
</p>

**Code with np:** 
```python
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```
**Scikit-Learn Code:** 
```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)
```
The Scikit-Learn implementation is based on `scipy.linalg.lstsq()` function, which computes the pseudoinverse of X calculated with the Singular Value Descomposition (SVD) technique. This approach is more efficient than calculating the Normal Equation (this method can not handle non-invertible matrices) because the pseudoinverse is always defined and manage edges cases nicely. 

The computational complexity of Normal equation is O(n^3), while the correspondinf value for SVD is O(n^2). Both strategies run very slow when the numer of features is huge (~100.000), but they are linear with reagard to the number of instances in the training set. 

### **Gradient Descent (GD)**

GD is a **generic optimization algorithm** capable of finding optimal solutions to a wide range of problems. In general, this method tries to **tweak parameters iteratively in order to minimize the cost function**. 

The algorithm measures the local gradient of the error function with regard to the parameter vector, and it goes in the direction of descending gradient. If the gradient is zero, the minimum was reached. 

Usually, the parameter vector initiates with random values, the cost function is minimized until the convergence to a minimum. The **learning rate** hyperparameter or the size of steps is important to consider in the GD. 

One aspect worth to consider is the **distribution of the cost function**, which could be irregular, so the GD would get stuck in local minimums instead of the desired global minimum. The good news is that the MSE cost function for LR is a **convex function**, which means that there are no local minima, just one global minimum. So, the GD is guaranteed to approach close to the global minimum. 

When using GD, you should assure that all features have **similar scale**. Otherwise, it will take much longer to converge. The GD is suited for cases where there are large numbers of features or training instances. 

#### **Batch GD** 

To implement the GD, we need to compute the gradient of the cost function with regard to each model parameter, or in other words how much the cost function will change if we tweak each parameter a little bit. This process can be obtained calculating the **partial derivatives**. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{\delta}{\delta\theta_{j}}MSE\left(\Theta\right) = \frac{2}{m} \sum_{i=1}^{m}\left(\Theta^{T}x^{\left(i\right)}-y^{\left(i\right)}\right)x_{j}^{\left(i\right)}"/>
</p>

It is possible to calculate the partial derivatives of all the parameters at time using a **gradient vector**, which contains all the partial derivatives of the cost function. In this way, the BGD uses the whole batch of training data at every step, which makes this implementation slow on large training datasets. But, the BGD scales well with the number of features. 

The gradient vector points uphill, so we need to substract this vector from the parameter vector times the learninng rate, in order to determine the size of the downhill step. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \Theta^{\left(next step\right)} = \Theta - \eta\nabla _{\Theta}MSE\left(\Theta\right)"/>
</p>

**Code:** 
```python
eta = 0.1 # learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)
 # random initialization
for iteration in range(n_iterations):
gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
theta = theta - eta * gradients
```
It is possible to use grid search to find a suitable learning rate. Regarding to the number of iterations to run the algorithm, a good startegy is set a large number of iterations but interrupt the algorithm when the gradient vector becomes tiny, which can be established with a value of **tolerance**. 

#### **Stochastic GD** 

This algorithm picks a random instance in the training set at every step and computes the gradients based only on a single instance. In this way, the SGD is faster and allows to train huge datasets, since one instance need to be in memory at each iteration. 

However, the SGD is less regular than the BGD and it will not reach the global minimum, only a site close to this point. So, **the final parameter values are good, but not optimal**. If the cost function is irregular, the SGD has a better chance of finding the global minimum than then BGD because randomness helps to escape from local optima. However, this algorithm can never settle at the minimum, which can be solved by gradually reducing the learning rate (**simulated annealing**). The function that determines the learning rate at each iteration is the **learning schedule**. 

A good practice is shuffling the instances during training, in such a way that instances are independent and identically distributed, to ensure that the parameters get pulled toward the global optimum, on average. 


**Code with np:** 
```python
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
def learning_schedule(t):
return t0 / (t + t1)
theta = np.random.randn(2,1)
 # random initialization
for epoch in range(n_epochs):
for i in range(m):
random_index = np.random.randint(m)
xi = X_b[random_index:random_index+1]
yi = y[random_index:random_index+1]
gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
eta = learning_schedule(epoch * m + i)
theta = theta - eta * gradients
```

**Scikit-Learn Code:** 
```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
```

#### **Mini-batch GD** 
This algorithm computes the gradients on small random sets of instances called **mini-batches**. The advantage of the MBGD over the SGD is that it has better performance on hardware matrix operations. Also, its progress in parameter space is less erratic than with SGD, especially with fairly large mini-batches. In this way, MBGD ends up a bit closer to the minimum than SGD.

It is worth noting that there is almost no difference after training because all the GD algorithms make predictions in exactly the same way. 

### **Polynomial Regression**

This algorithm allows to apply a linear model to fit nonlinear data. This can be done adding powers of each features as new features, and train a linear model on this extended set of features. 

It is possible to tranform the training data, adding powers of features to the dataset. 

**Scikit-Learn Code:** 
```python
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
```

This model is capable of finding relationships between features. 

### **Learning Curves**

If a model performs well on the training data but
generalizes poorly according to the cross-validation metrics, then your model is
**overfitting**. If it performs poorly on both, then it is **underfitting**.

The **learning curves** are plots of the
model’s performance on the training set and the validation set as a function of
the training set size (or the training iteration).

**Scikit-Learn Code:** 
```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
```
At some point, the error eaches a plateau, at which point adding new instances does not make the average error much better or worse. If there is a gap between both curves, and the model performs better on the training data, it can be a signal of an overfitting model. 

If the model is underfitting, it is required to use a more complex model or obtain better features. In the other hand, if te model is overfitting a possible solution would be add more trainig data until the validation error reaches the training error. 

A model's generalization error can be expressed as the sum of three different errors: 

1. **Bias:** this part of the error is due to wrong assumptions. A high bias model is likely to underfit the training data. 
2. **Variance:** this part is due to the model's excessive sensitivity to small variations in the training data. A high variance model is prone to overfitting. 
3. **Irreducible error:** this part is due to noiseness of the data itself. The only way to reduce this error is to clean up the data. 

### **Regularized Linear Models**

For a linear model, regularization is typically achieved by constraining the weights of the model. 

#### **Ridge Regression**

In this model, a regularization term **α** is added to the cost function. The α should only be added to the cost function during training, while an unregularized performance measure is applied to evaluate the model's performance. If alpha is equal to 0 this model is just LR. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large J\left(\Theta\right)= MSE\left(\Theta\right) %2B \alpha\frac{1}{2}\sum_{i=1}^{n}\Theta_{i}^{2}"/>
</p>

This model is associated with the l2 norm. It is important to scale the data before performing Ridge
Regression, as it is sensitive to the scale of the input features. Increasing α leads
to flatter predictions, thus reducing the
model’s variance but increasing its bias. 

It is possible to compute RR either by
computing a closed-form equation or by performing GD.

**Scikit-Learn Code:** 
```python
from sklearn.linear_model import Ridge

# Closed form 
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

# SGD
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```

#### **Lasso Regression**

This model also adds a regularization term to the cost function, but it uses l1 norm of the weight vector instead of the l2 norm. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large J\left(\Theta\right)= MSE\left(\Theta\right) %2B \alpha\sum_{i=1}^{n}\left|\Theta_{i}\right|"/>
</p>

This model tends to eliminate the
weights of the least important features, so it  automatically performs feature
selection and outputs a sparse model. To avoid Gradient Descent from bouncing around the optimum at the end when using Lasso,
you need to gradually reduce the learning rate during training 

**Scikit-Learn Code:** 
```python
from sklearn.linear_model import Lasso

# Closed form 
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])

# SGD
sgd_reg = SGDRegressor(penalty="l1")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```

#### **Elastic Net**

This model is a middle ground between Ridge Regression and Lasso Regression.
The regularization term is a simple mix of both Ridge and Lasso’s regularization
terms. When r = 0, Elastic Net is equivalent
to Ridge Regression, and when r = 1, it is equivalent to Lasso Regression. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large J\left(\Theta\right)= MSE\left(\Theta\right) %2B r\alpha\sum_{i=1}^{n}\left|\Theta_{i}\right| %2B \frac{1-r}{2}\alpha\sum_{i=1}^{n}\Theta_{i}^{2} "/>
</p>

It is almost always preferable to
have at least a little bit of regularization, so generally you should avoid plain
Linear Regression. Ridge is a good default, but if you suspect that only a few
features are useful, you should prefer Lasso or Elastic Net because they tend to
reduce the useless features’ weights down to zero. Elastic Net is preferred over Lasso because Lasso may behave
erratically when the number of features is greater than the number of training
instances 

**Scikit-Learn Code:** 
```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```

#### **Early Stopping**

This is a different way to regularize iterative learning algorithms such as Gradient
Descent is to stop training as soon as the validation error reaches a minimum. With early stopping you just stop training as soon as the validation
error reaches the minimum.

**Scikit-Learn Code:** 
```python
from sklearn.base import clone

# prepare the data
poly_scaler = Pipeline([
	("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
	("std_scaler", StandardScaler())
])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
						penalty=None, learning_rate="constant", eta0=0.0005)
						
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
	sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
	y_val_predict = sgd_reg.predict(X_val_poly_scaled)
	val_error = mean_squared_error(y_val, y_val_predict)
	if val_error < minimum_val_error:
		minimum_val_error = val_error
		best_epoch = epoch
		best_model = clone(sgd_reg)
```