# NotesDay24- Machine Learning Zoomcamp sixth week

## 6.5 Decision trees parameter tuning

Decision trees have multiple parameters, such as **max depth**, which defines how deep the tree can grow or number of layers in the tree. Another parameter is the **minimum size of a group**, a criteria to decide if one side or group of a tree is sufficiently large. Decision trees have more parameters, but in this session we will cover only the two mentioned before.

Parameter tuning means to choose parameters in such a way that model's performance or evaluation metric is maximized or minimized, depending on what metric we choose. For this project, we will try to maximize AUROC metric.

A good strategy to tune various parameters in large datasets is first find the best values from one parameter, and then combine them with other values from other parameters.

**Classes and methods:**

* `DecisionTreeClassifier(max_depth=z, min_samples_leaf=w).fit(x,y)` - Scikit-Learn class to train a decision tree classifier with x feature matrix and y target values, using z as the maximum depth and w as minimum size of the left group.
* `df.pivot(index='x', columns='y', values='z')` - pandas method to transform the structure of a dataframe df considering x as rows, y as columns, and z as cell values.
* `sns.heatmap(df, annot=True, fmt='.zf')` - seaborn method to create a heatmap from a df dataframe with annotations and z rounding values.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).

## 6.6 Ensemble learning and random forest

In general, ensemble models use various models and aggregate their results with some operator such as mean, maximum, or another one. So, prediction of ensemble models is the aggregated operator of results from all models. RF is an ensemble of independent decision trees, and each of these models gets a random subset of features.

There is a point at which RF does not improve its performance although we increase number of trees in the ensemble. We also fine-tune **max depth**,and **minimum size of a group** because RF is a bunch of decision trees.

Other interesting parameters to tune are **the maximum number of features** and **bootstrap**, which is a different way of randomization at the row level. Also, **n_jobs** parameter allows us to train models with parallel processing.

**Classes and methods:**

* `RandomForestClassifier(n_estimators=z, random_state=w).fit(x,y)` - Scikit-Learn class to train a random forest classifier with x feature matrix and y target values, using z decision trees and a random seed of w.
* `dt.predict_proba(x)[:,1]` - predict x values with a dt Scikit-Learn model, and extracts only the positive values.
* `zip(x, y)` - function that takes x and y iterable or containers and returns a single iterator object, having mapped values from all the containers.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).

## 6.7 Gradient boosting and XGBoost

Boosting algorithms are a type of ensemble models in which each model takes as input results of previous models in a sequential manner. At each step of boosting, the aim is to reduce errors of previous models. Final prediction of boosting algorithms consider predictions of all models using an aggregated operator. When models in boosting are decision trees, we called it as **Gradient boosting trees (GBT)**.

A good library for working with GBT is **xgboost**. Some important parameters of GBT from xgboost are:

* **eta:** learning rate, which indicates how fast our model learns
* **max_depth:** depth size of trees
* **min_child_weight:** how many observation we have in a leave node
* **objective:** type of model we will apply
* **eval_metric:** specify evaluation metric to use
* **nthread:** parallelized training specifications
* **seed:** random seed
* **verbosity:** type of warnings to show

**Libraries, classes and methods:**

* `xgboost` - Python library  that implements machine learning algorithms under the Gradient Boosting framework.
`xgb.DMatrix(x, label=y, feature_names=z)` - xgboost method for converting data into DMatrix, a special structure for working on training data with this library. We need to provide an x feature matrix, a y target vector, and a z vector of features names.
* `xgb.train(x, dtrain, num_boost_round=y, evals=z)` - xgboost method for training a dtrain DMatrix with x dictionary of parameters, y number of trees, and a z watchlist to supervise model performance during training.
* `x.predict(y)` - method from an x xgboost model to make predictions on a y validation dataset.
* `%%capture` - Jupyter notebook command to capture everything a code cell outputs into a string.
* `s.split('x')` - string method to split it by x separator.
* `s.strip('x')` - string method to delete some x characters.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).

## 6.8 XGBoost parameter tuning

**Learning rate** or **eta** refers to size of step in **Gradient boosting trees (GBT)**, and it tell us how fast a model learns. In other words, eta controls how much weight is applied to correct predictions of a previous model.

It is recommended to tune first eta parameter, then max_depth, and min_child_weight at the end. Other important parameters to consider for tuning are **subsample**, **lambda**, and **alpha**.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).

## 6.9 Selecting the best model

The best tree based model for credit risk scoring project was **Gradient boosting trees (GBT)**. So, we trained this model on the entire training dataset, and evaluated it on test dataset.

Usually, GBT is one of the models with the best performance for tabular data (dataframe with features). Some downsides fo GBT models are their complexity, difficulty for tunning, and tendency to overfitting.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).

## 6.10 Summary

* Decision trees have nodes with conditions, where we split the dataset and we have decision nodes with leaves when we take a decision. These models learn if-then-else rules from data, and they tend to overfit if we do not restrict them in terms of depth growing.
* For finding the best split of decision trees, we select the least impure split. This algorithm can overfit, that's why we control it by limiting the max depth and the size of the group.
* Random forest is a way of combining multiple decision trees in which each model is trained independently with a random subset of features. It should have a diverse set of models to make good predictions.
* Gradient boosting trains model sequentially: each model tries to fix errors of the previous model. XGBoost is an implementation of gradient boosting.
