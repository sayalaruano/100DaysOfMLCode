# NotesDay15 - Machine Learning Zoomcamp fourth week 

## 4.5 ROC curves
ROC stands for Receiver Operating Characteristic, and this idea was applied during the Second World War for evaluating the strenght of radio detectors. This measure considers **False Positive Rate** (FPR) and **True Postive Rate** (TPR), which are derived from the values of the confusion matrix.

**FPR** is the fraction of false positives (FP) divided by the total number of negatives (FP and TN - the first row of confusion matrix), and we want to minimize it. The formula of FPR is the following: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{FP}{TN %2B FP}"/>
</p>

In the other hand, **TPR** or **Recall** is the fraction of true positives (TP) divided by the total number of positives (FN and TP - second row of confusion table), and we want to maximize this metric. The formula of this measure is presented below: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B FN}"/>
</p>

ROC curves consider Recall and FPR under all the possible thresholds. If the threshold is 0 or 1, the TPR and Recall scores are the opposite of the threshold (1 and 0 respectively), but they have different meanings, as we explained before. 

We need to compare the ROC curves against a point of reference to evaluate its performance, so the corresponding curves of random and ideal models are required. It is possible to plot the ROC curves with FPR and Recall scores vs thresholds, or FPR vs Recall. 


**Classes and methods:** 
* `np.repeat([x,y], [z,w])` - returns a numpy array with a z number of x values, and a w number of y values. 
* `roc_curve(x, y)` - sklearn.metrics class for calculating the false positive rates, true positive rates, and thresholds, given a target x dataset and a predicted y dataset. 

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/04-evaluation/notebook.ipynb). 


## 4.6 AUROC

The Area under the ROC curves can tell us how good is our model with a single value. The AUROC of a random model is 0.5, while for an ideal one is 1. 

In ther words, AUC can be interpreted as the probability that a randomly selected positive example has a greater score than a randomly selected negative example.

**Classes and methods:** 

* `auc(x, y)` - sklearn.metrics class for calculating area under the curve of the x and y datasets. For ROC curves x would be false positive rate, and y true positive rate. 
* `roc_auc_score(x, y)` - sklearn.metrics class for calculating area under the ROC curves of the x false positive rate and y true positive rate datasets.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/04-evaluation/notebook.ipynb).  

## 4.7 Cross-validation

Confusion table is a way to measure different types of errors and correct decisions that binary classifiers can made. Considering this information, it is possible evaluate the quality of the model by different strategies.

If we predict the probability of churning from a customer, we have the following scenarios:

* No churn - **Negative class**
    * Customer did not churn - **True Negative (TN)**
    * Customer churned - **False Negative (FN)**
* Churn - **Positive class**
    * Customer churned - **True Positive (TP)**
    * Customer did not churn - **False Positive (FP)**

The confusion table help us to summarize the measures explained above in a tabular format, as is shown above: 

|**Actual/Predictions**|**Negative**|**Postive**|
|:-:|---|---|
|**Negative**|TN|FP|
|**Postive**|FN|TP| 

The **accuracy** corresponds to the sum of TN and TP divided by the total of observations. 

The code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/04-evaluation/notebook.ipynb).  

## 3.4 Precision and Recall
**Precision** tell us the fraction of positive predictions that are correct. It takes into account only the **positive class** (TP and FP), as is stated in the following formula:

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B FP}"/>
</p>

**Recall** measures the fraction of correctly identified postive instances. It considers parts of the **postive and negative classes** (TP and FN). The formula of this metric is presented below: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TP}{TP %2B FN}"/>
</p>

 In this problem, the precision and recall values were 67% and 54% respectively. So, these measures reflect some errors of our model that accuracy did not notice due to the **class imbalance**. 