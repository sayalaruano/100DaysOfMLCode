# NotesDay23- Machine Learning Zoomcamp sixth week

## 6.1 Credit risk scoring project

Sixth week of Machine Learning Zoomcamp is about Decision trees and Ensemble learning. We will work in a project of credit risk scoring. So, the aim is creating a model that given some bank customers, it predicts the risk that customers are not going to pay back an amount of lend money, or in other words that they are going to default. This model can help banks to take the decision if they should lend money for their customers or not.

This model will be a binary classifier that outputs a probability of default. If this probability to lower than a cutoff, it is assigned to 0, which means that customers will probably pay back lend money. In other hand, if the probability is greater than the threshold, it is assigned to 1, an it is likely that customers will default. We can use historical data and customers' applications as features of the model.

The dataset for this project is available [here](https://github.com/gastonstat/CreditScoring), and the entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).

## 5.2 Data cleaning and preparation

In this session, we downloaded the dataset, renamed categorical variables, replaced missing values, and did the train/validation/test split.

**Commands and methods:**

* `wget x` - Unix command to download an x file providing its URL.  
* `head x` - Unix command to show the first lines of an x file.
* `pd.read_csv(x)` - pandas method for loading an x csv file.
* `df.columns.str.lower()` - pandas method for lowercase all the letters in the names of columns from a df dataframe.
* `df.x.value_counts()` - count the number of instances in each category of x column from a df dataframe.
* `df.x.map(y)` - change the values of x column with the information of y dictionary from a df dataframe.
* `df[x].replace(to_replace=y, value=z)` - replace y values of x column with z values from a df dataframe.
* `df[df.x != 'y'].reset_index(drop=True` - delete rows equal to y of x column, and make the indices sequential from a df dataframe.
* `df[x].replace(to_replace=y, value=z)` - replace y values of x column with z values from a df dataframe.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).

## 5.3 Decision trees

A decision tree is a data structure composed of nodes that represent conditions, which can be true or false, and at the final node we have to take a decision. Basically, decision trees are made by a bunch of chained if-else conditions.

Simple decision trees are prone to overfitting, which means that the model can not generalize well, and it only memorizes patterns from the training set. The tendency of decision trees to overfit can be associate with the specific rules followed by these models. In general, deeper trees tend to overfit data. So, we can avoid overfitting by restricting depth of trees. But, if we restrict too much the tree until only one level (decision stump), it is a very simple model.

**Classes and methods:**

* `df.to_dict()` - pandas method to transform a df dataframe to a dictionary.
`DictVectorizer().fit_transform(x)` - Scikit-Learn class for converting x dictionaries into a sparse matrix, and in this way doing the one-hot encoding. It does not affect the numerical variables.
* `DecisionTreeClassifier().fit(x,y)` - Scikit-Learn class to train a decision tree classifier with x feature matrix and y target values.
* `dt.predict_proba(x)` - predict x values with a dt Scikit-Learn model.
* `export_text(x, feature_names=y)` - Scikit-Learn class to print x tree with y feature names.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).

## 5.4 Decision tree learning algorithm

A decision tree is made of **decision nodes** that can be true or false, taking into account if a the value of a feature is greater or equal to a threshold. Also, there are leaves, which  are the lowest nodes in a tree.

A key part of decision tree models are thresholds chose to compare features values in decision nodes. This decision can be made by different split evaluation criteria such as **misclassification rate**, a way to calculate **impurity**, which measures proportion of predictions' errors with respect to the expected decision. We need to consider the average impurity among left and right decision nodes. The best models aim to have the lowest impurity values.

In brief, finding the best split algorithm can be summarized as follows:

* FOR F IN FEATURES:
    * FIND ALL THRESHOLDS FOR F:
        * FOR T IN THRESHOLDS:
            * SPLIT DATASET USING "F>T" CONDITION
                * COMPUTE IMPURITY OF THIS SPLIT
* SELECT THE CONDITION WITH THE LOWEST IMPURITY

The algorithm can iterate until it completes all possible splits, but we can stablish a stopping criteria to define when we need to stop the splitting process. The stopping condition can be defined considering criteria such as:

* Group already pure
* Tree reached depth limit
* Group too small to split

Decision tree learning algorithm can be summarized as:

1. Find the best split
2. Stop if max-depth is reached
3. If left is sufficiently large and not pure
    * Repeat for left
4. If right is sufficiently large and not pure
    * Repeat for right

Decision trees can be used for regression tasks.

**Libraries, classes and methods:**

* `df.sort_values('x')` - sort values of an x column from a df dataframe.
* `display()` - IPython.display class to print values inside of a for-loop.

The entire code of this project is available in [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/notebook.ipynb).
