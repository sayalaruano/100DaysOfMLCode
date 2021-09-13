# NotesDay3

## Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow chapter 2 - End-to-End Machine
Learning Project

### Look at the big picture 
#### Frame the Problem
This part refers to understand the business or research objective, which is a key step because 
the selection of model, performance masure, among other aspects depends on the problem intented to solve. 
**Pipeline:** A sequence of data processing components 
It is important to consider what is the current solution to solve the problem, so this can serve as a framework to comparing the ML model.
**Multiple regression problem:** model trained with multiple features to make the pediction. 
**Univariate regression problem:** predicts a single value. 

#### Select a Performance Measure
These functions are ways to measure the distance between the vector of predictions and the vector of target values. 
**Root Mean Square Error (RMSE):** cost function that provides an idea of the error made by the systems in its predictions - higher weight for large errors.
It uses the *Euclidian norm* or *l2 norm* for calculating the distance. 
**Mean Absolute Error (MSE)**If there are many outliers in the dataset, the **mean absolute error (MSE)** is a prefered cost function. It uses the 
*Manhattan norm* or *l1 norm* for caculating the distance, which measures the distance between two points only if these are orthogonal 
The higher the norm index, the more it focuses on large vaules and neglects the samll ones. 

#### Check the Assumptions
This can be applied depending on the model requirements. 

### Get the Data

#### Create the Workspace
It is recommended to create an isolated environment for each project. 

#### Download the Data
Some libraries for this task: os, tarfile, urllib
**Code:** 
```python
import tos
impor tarfile
impor urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	os.makedirs(housing_path, exist_ok=True)
	tgz_path = os.path.join(housing_path, "housing.tgz")
	urllib.request.urlretrieve(housing_url, tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()
```
#### Take a Quick Look at the Data Structure
Pandas methods: head(), info(), value_counts(), describe(), hist(), 

#### Create a Test Set
**Data snooping bias:** estimate the generalization error using the test set. 
**In house function:**
```python
import numpy as np
def split_train_test(data, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]
```
It is possible to define a random seed for obtaining always the same shuffled indices - np.random.seed(..)

Also, a hash identifier for each instance can be calculated to ensure that the test set will remain consistent across multiple
runs, although the dataset is changed. 

**In house function with hash identifiers:**
```python
from zlib import crc32

def test_set_check(identifier, test_ratio):
	return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
	return data.loc[~in_test_set], data.loc[in_test_set]
```

**Sklearn function:**
```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```
Also, depending on the dataset, it could be a requirement to make a **stratified sampling**, in which the dataset is divided in 
homogenous groups and the right number of instances are put on the test dataset, guaranteeing the representativeness of the overall population. 

**Sklearn function:**
```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]
```
### Discover and Visualize the Data to Gain Insights

#### Looking for Correlations
The correlation can be calculated between pairs of attributes or one feature and the target variable. 

Some functions: corr(), scatter_matrix()

**Code:**
```python
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```
#### Experimenting with Attribute Combinations
This can be useful to obtain better features and improve the model. 

### Prepare the Data for Machine Learning Algorithms
It is useful to use functions or pipelines for this part, so we can reproduce the processing with other datasets. 

#### Data Cleaning
This is mainly refered to missing values. There are 3 options to handling this:
1. Get rid of the missing values
2. Get rid of the whole feature 
3. Set the values to some value (zeo, mean, median, etc)

Pandas methods: dropna(), drop(), fillna()
Sklearn class: SimpleImputer 

**Sklearn code:**
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```
**Sklearn design:**
* Estimators: estimate some parameters based on a dataset - fit() method
* Transformers: transform a dataset - transform() method or fit_transform() method
* Predictors: making predictions - predict() method 
* Inspection: estimator’s hyperparameters are accessible directly via public instance variable, while the learned ones have an underscore suffix

#### Handling Text and Categorical Attributes
**Sklearn classes:** OrdinalEncoder, OneHotEncoder
**Embedding:** represent a categorical attribute as learnable and low dimensional vectors

#### Custom Transformers
Scikit-Learn relies on duck typing (not inheritance), which refers to  create a class and implement
three methods: fit() (returning self), transform(), and fit_transform(). 

#### Feature Scaling
* Min-max scaling or normalization: values rescaled so that they end up ranging from 0 to 1, by subtracting the min value and 
dividing the max minus the min -  MinMaxScaler sklearn's class. 
* Standarization: it subtracts the mean value and divide by the standard deviation. It does not bound values to a specific range, 
but it is less affected by outliers -  StandardScaler sklearn's class. 

#### Transformation Pipelines
**Pipeline sklearn's class:** It takes a list of name/estimator pairs defining a sequence of steps. All but the last estimator must be transformers.
When the pipeline’s fit() method is used, it calls fit_transform() sequentially on all transformers. 
**ColumnTransformer sklearn's class:** allow to handle certain columns with different trasnformers. 

### Select and Train a Model
#### Training and Evaluating on the Training Set
**Sklearn code:**
```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```
The DecisionTreeRegressor algorithm can be used for finding complex nonlinear relationships in the data. 

#### Better Evaluation Using Cross-Validation
**Scikit-Learn’s K-fold cross-validation feature:** splits the training set in n folds, then train and evaluate the model n times, 
picking a different fold for evaluation every time and training ont the remaining folds. 
Cross-validation allows to get not only an estimate of the performance of your model, but also a measure of 
how precise this estimate is (i.e., its standard deviation). But this method comes at the cost of training the model several times, 
so it is not always possible.
**Sklearn code:**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```
**RandomForestRegressor:**  training many Decision Trees on random subsets of the features, then averaging out their predictions - **Ensemble learning**
(try to combine the models that perform best).

### Fine-Tune Your Model

#### Grid Search
**Scikit-Learn’s GridSearchCV:** train models with different combinations of hyperparameters, and evaluate them with cross-validation. 

**Sklearn code:**
```python
from sklearn.model_selection import GridSearchCV
param_grid = [
	{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
	{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
							scoring='neg_mean_squared_error',
							return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```
It is possible to treat some of the data preparation steps as hyperparameters. 

#### Randomized Search
**Scikit-Learn’s RandomizedSearchCV:** instead of trying out all possible combinations, it evaluates a given number of 
random combinations by selecting a random value for each hyperparameter at every iteration. 

#### Analyze the Best Models and Their Errors

#### Evaluate Your System on the Test Set
Evaluation of the final model on the test set. For this, we need to get the predictors and
the labels from your test set, run your full_pipeline to transform the data (call transform(), not fit_transform()—you do not want to fit the test set!),
and evaluate the final model on the test set. 

### Launch, Monitor, and Maintain Your System
Deploy the final model to a production environment, which can be located on a web server or on the cloud. 

An example of a library for exporting a model is *joblib*. 

It is important to have monitoring code to check the system's performance over time. 














