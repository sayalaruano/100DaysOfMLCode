# **NotesDay17**

## **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow chapter 3 - Classification**

### **MNIST Dataset**

**MNIST** is a dataset of 70,000
small images of digits handwritten by high school students and employees
of the US Census Bureau. Each image is 28x28 pixels. 

Scikit-Learn has various helper functions to download popular datasets, including the MNIST dataset. Then, we can show one of the images using matplotlib. 

**Code:** 
```python
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt

# Download dataset
mnist = fetch_openml('mnist_784', version=1)

# Print components of the dataset
mnist.keys()

# Save the X fature matrix and y target values 
X, y = mnist["data"], mnist["target"]

# Convert y values to int
y = y.astype(np.uint8)

# Show one of the images 
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image,plt.axis("off")
plt.show()

# Separate the test partition 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

### **Training a Binary Classifier** 

The **Stochastic Gradient Descent (SGD) classifier** can be used for binary classification tasks. SGDC deals with training instances independently, one at a time, and it can handle large datasets efficiently. 

**Code:** 
```python
from sklearn.linear_model import SGDClassifier

# Create and train a SGDC 
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Make predictions 
sgd_clf.predict([some_digit])
``` 

### **Performance Measures**

#### **Measuring Accuracy Using Cross-Validation**

An option to evaluate a model is using **cross-validation**. **K-
fold cross-validation** means splitting the training set into K folds, then making predictions and evaluating them on each fold
using a model trained on the remaining folds. It is possible to employ the classes available in Scikit-Learn, but we can also implement a our own function to have more control over this algorithm. 

**Code:** 
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# Create the Satratified K-fold object
skfolds = StratifiedKFold(n_splits=3, random_state=42)

# Perform the K-fold algorithm
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) 
```

The **StratifiedKFold** class performs stratified sampling to produce folds that contain a representative ratio of each
class. At each iteration the code creates a clone of the classifier, trains
that clone on the training folds, and makes predictions on the test fold.

To evaluate the SGDC with accuracy by cross validation algortihm, we can use the following code: 

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```
**Accuracy** is not the ideal metric to evaluate a classifier, especially when the dataset is skewed or unbalanced. This is because if the dataset has more instances of one class than the other one, the predictions will be more likely to be right for the predominant class.

#### **Confusion Matrix**

The confusion matrix is constructed by counting the number of times instances of class A are classified as class B. We can make the predictions and create the confusion matrix with the following code: 

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

confusion_matrix(y_train_5, y_train_pred)
```
A summary of the confusion matrix is shown below: 

|**Actual/Predictions**|**Negative**|**Postive**|
|:-:|---|---|
|**Negative**|TN|FP|
|**Postive**|FN|TP| 

Each row represents an **actual class**, while each column is a **prediction class**. Depending on the predictions, we have TN, FN, TP, and FP. A perfect classifier would have only TN and TP, so its matrix would have non-zero values only on it main diagonal. 

#### **Precision and Recall**

**Precision** is the accuracy of the positive predictions. It takes into account only the **positive class** (TP and FP - second column of the confusion matrix), as is stated in the following formula:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B FP}"/>
</p>

**Recall**, also called **sensitivity** or **true positive rate (TPR)**, is the ratio of positive instances that are correctly detected by the classifier.  It considers parts of the **postive and negative classes** (TP and FN - second row of confusion table). The formula of this metric is presented below: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B FN}"/>
</p>

**F1 score** combines precision and recall into a single metric. It is the **harmonic mean** of precision and recall, which gives more weight to low values. Consequently, the classifier get a high F1 score if both recall and precision are high, so this measure favors classifiers that have similar precision and recall. The formula of this metric is presented below: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B \frac{FN %2B FP}{2}}"/>
</p>

Scikit-Learn provides some classes to calculate these metrics: 

```python
from sklearn.metrics import precision_score, recall_score, f1_score 

precision_score(y_train_5, y_train_pred) 

recall_score(y_train_5, y_train_pred) 

f1_score(y_train_5, y_train_pred)
```

#### **Precision/Recall trade-off** 

This trade-off is associated with the fact thta increasing precision reduces recall, and vice versa. The PR curves show us this pattern. 

**Code:**
```python
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
```

It is important to analyze of all of these measures because a high-precision classifier is not very useful if its recall is too low. 

#### **The ROC courve**

The Receiver Operating Characteristic (ROC) curve plots the **False Positive Rate** (FPR) against the **True Postive Rate** (TPR), which are derived from the values of the confusion matrix.

**FPR** is the fraction of false positives (FP) divided by the total number of negatives (FP and TN - the first row of confusion matrix), and we want to minimize it. The formula of FPR is the following: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{FP}{TN %2B FP}"/>
</p>

In the other hand, **TPR** or **Recall** is the fraction of true positives (TP) divided by the total number of positives (FN and TP - second row of confusion table), and we want to maximize this metric. The formula of this measure is presented below: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B FN}"/>
</p>

Scikit-Learn provides a class to calculate the TPR and FPR, and with this data we can plot the ROC curves, applying the following code:

```python
from sklearn.metrics import roc_curve

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') 

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plot_roc_curve(fpr,plt.show()
tpr)
```

A good classifier stays as far away as possible from the diagonal of the plot (toward the top-left corner), which corresponds to the performance of a random classifier. 

We can compare classifiers using the **area under the ROC curve (AUROC)**. A perfect classifier will have an AUCROC equal to 1, whereas a
purely random classifier will have an AUCROC equal to 0.5. Scikit-Learn
provides a function to compute this metric:

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
```

### **Multiclass Classification**

Multiclass classifiers can distinguish between more than two classes. There are two strategies to create this type of classifier: 
* **One-versus-the-rest (OvR) strategy:** Create binary classifiers for each class, and select the class whose classifier outputs the hightest score.  
* **One-versus-one (OvO) strategy:** Create binary classifiers for every pair of classes. The main advantage of this method is that each classifier only need to be trained on the part of the training set for the two classes that it must distinguish. 

Some algorithms (such as Support Vector Machine classifiers) scale poorly
with the size of the training set. For these algorithms OvO is preferred
because it is faster to train many classifiers on small training sets than to
train few classifiers on large training sets. For most binary classification
algorithms, however, OvR is preferred.

If you want to force Scikit-Learn to use one-versus-one or one-versus-the-
rest, you can use the OneVsOneClassifier or OneVsRestClassifier
classes, as is shown below: 

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
```

### **Multilabel Classification**

The **multilabel classifiers** output multiple classes for each instance, such as the face recognition algorithms. 

### **Multioutput Classification**

These classifiers are a generalization of multilabel classifiers where each label can be multiclass. The classifier’s output is multilabel and each label can have multiple values. 