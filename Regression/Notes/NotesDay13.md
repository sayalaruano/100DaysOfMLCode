# **NotesDay10**

## **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow chapter 4 - Training Models**

### **Logistic regression**

The Logistic Regression algorithm estimates the probability that an instance belongs to a particular category. If the estimated probability is greater than 50%, then the model predicts that the record belongs to that class (positive class, labeled as 1), and otherwise that it does not (negative class, labeled as 0). So, Log Reg is a binary classifier. 


This model works similarly than the Linear Regression model, calculating the weighted sum of the input features (plus a bias term), but the output is the **logistic** of this sum. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \hat{y} = h_{\Theta}\left( x \right) = \sigma\left(\Theta \cdot x\right)"/>
</p>

The logistic is the **sigmoid function** that outputs a number between 0 and 1. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \sigma\left(\t\right)=\frac{1}{1%2Bexp\left( -t \right)}"/>
</p>

If the calculated probability is greater than or equal to 0.5, the y prediction would be 1, or 0 otherwise. Also, it can be noticed that the probability is less than 0.5 if *t* is a negative number, and this probability is greater than or equal to 0.5 if *t* is a positive number. 

#### **Training and cost function** 

THe cost function of Logistic Regression is the **log loss**, which allows to distinguish between positive and negative predictions.  

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large J\left(\Theta\right)= -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}log\left(\hat{p}^{(i)}\right) %2B \left(1-y^{(i)}\right) log\left(1-\hat{p}^{(i)}\right)\right]"/>
</p>

There is no known clsed-form equation to compute the value of tetha that minimizes the cost function. But, the log loss is a convex function, so Gradient Descent is guaranteed to find a global minimum, so we can use any variation of GD to find the weights that minimizes the cost function.  

### **Decision boundaries**

**Code:** 
```python
from sklearn.linear_model import 

LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)


plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")
```
With this code applied to the iris dataset, there is a decision boundary where both probabilities are equal to 50%. This point represents the borderline for the model to make a prediction. 

Logistic regression can be regularized using l1 or l2 penalties. The Scikit-Learn function adds an l2 penalty by default. The hyperparameter that controls the model regularization is C, which corresponds to the inverse of alpha. The higher the value of C, the less the model is regularized.  

#### **Softmax Regression** 

This algorithm is a generalization of the Logistic regression model that supports **multiple classes** directly, without having to train and combine multiple binary classifiers. The idea behind this model is that a score is computed for each class, then estimates the probability of each class by applying the **softmax function**. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \hat{p_{k}} = \sigma\left(s\left(x\right)\right)_{k} = \frac{exp\left(s_{k}\left(x\right)\right)}{\sum_{j=1}^{K}exp\left(s_{j}\left(x\right)\right)}"/>
</p>

The softmax regression classifier predicts the class with the hightest estimated probability, which is calculated using the **argmax** operator. 

The cost function for this algorithm is the **cross entropy**, which penalizes the model when it estimates a low probability for a target class. This metric is used to measure how well a set of estimated class probabilities matches the target classes. This cost function comes from information theory, and measures the average number of bits per option. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large J\left(\theta\right) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}y^{\left(i\right)}_{k}log\left(\hat{p}^{\left(i\right)}_{k}\right)"/>
</p>

Each class has its own parameter vector, which are stored as rows in a parameter matrix. Once you have computed the score of every class for the instance x, you can
estimate the probability that the instance belongs to class k by running the
scores through the softmax function. The scores are generally called **logits**. 

The same options of regularization for Logistic Regression are available for Softmax, and Scikit-Learn also applies l2 regularization by default. 

**Code:** 
```python
from sklearn.linear_model import 

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)

softmax_reg.predict([[5, 2]])
softmax_reg.predict_proba([[5, 2]])
```