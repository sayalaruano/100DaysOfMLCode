# NotesDay5 - Machine Learning Zoomcamp second week 

## 2.1 Car Price Prediction Project
This project is about the creation of a model for helping users to predict car prices. The dataset was obtained from [this 
kaggle competition](https://www.kaggle.com/CooperUnion/cardataset).

**Project plan:**

* Prepare data and Exploratory data analysis (EDA)
* Use linear regression for predicting price
* Understanding the internals of linear regression 
* Evaluating the model with RMSE
* Feature engineering  
* Regularization 
* Using the model 

The code and dataset are available at this [link](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-02-car-price). 

## 2.2 Data Preparation

**Pandas attributes and methods:** 

* pd.read.csv() - read csv files 
* df.head() - take a look of the dataframe 
* df.columns - retrieve colum names of a dataframe 
* df.columns.str.lower() - lowercase all the letters 
* df.columns.str.replace(' ', '_') - replace the space separator 
* df.dtypes - retrieve data types of all features 
* df.index - retrive indices of a dataframe

## 2.3 Exploratory Data Analysis

**Pandas attributes and methods:** 

* df[col].unique() - returns a list of unique values in the series 
* df[col].nunique() - returns the numer of unique values in the series 
* df.isnull().sum() - retunrs the numer of null values in the dataframe 

**Matplotlib and seaborn methods:**

* %matplotlib inline - assure that plots are displayed in jupyter notebook's cells
* sns.histplot() - show the histogram of a series 
   
**Numpy methods:**
* np.log1p() - applies log transformation to a variable and adds one to each result 

Long-tail distributions usually confuse the ML models,  so the recommendation is to transform the target variable distribution to a normal one whenever possible. 

## 2.4 Setting up the validation framework

In general, the dataset is split into three parts: training, validation, and test. For each partition, we need to obtain feature matrices (X) and y vectors of targets. First, the size of partitions is calculated, records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices. 

**Pandas attributes and methods:** 

* df.iloc[] - returns subsets of records of a dataframe, being selected by numerical indices
* df.rest_index() - restate the orginal indices 
* del df[col] - eliminates target variable 

**Numpy methods:**
* np.arrange() - retruns an array of numbers 
* np.random.shuffle() - retturns a suffled array
* np.random.seed() - set a seed 

## 2.5 Linear regression

Model for solving regression tasks, in which the objective is to adjust a line for the data and make predictions on new values. The input of this model is 
the feature matrix and a y vector of predictions is obtained, trying to be as close as possible to the actual y values. The LR formula is the sum of the 
bias term (WO), which refers to the predictions if there is no information, and each of the feature values times their corresponding weights. We need to 
assure that the result is showed on the untransformed scale. 

## 2.6 Linear regression: vector form

The formula of LR can be synthesized with the dot product between features and weights. The feature vector includes the bias term with an x value of one. 
When all the records are included, the LR can be calculated with the dot product between feature matrix and vector of weights, obtaining the y vector of 
predictions. 

## 2.7 Training linear regression: Normal equation

Obtaining predictions as close as possible to y target values requires the calculation of weights from the general LR equation. The feature matrix does not 
have an inverse because it is not square, so it is required to obtain an approximate solution, which can be obtained using the **Gram matrix** 
(multiplication of feature matrix and its transpose). The vector of weights or coefficients obtained with this formula is the closest possible solution to 
the LR system.
