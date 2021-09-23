# NotesDay11 - Machine Learning Zoomcamp third week 

## 3.1 Churn Prediction Project

The project aims to identify customers that are likely to churn or stoping to use a service. Each costumer has a score associated with the probability of churning. Considering this data, the company would send email with disccounts or other promotions to avoid churning. 

The ML strategy applied to approach this problem is binary classification, which for one instance can be expressed as: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large g\left(x_{i}\right) = y_{i}"/>
</p>

In the formula, yi is the model's prediction and belongs to {0,1}, being 0 the negative value or no churning, and 1 the positive value or churning. The output corresponds to the likelihood of churning. 

In brief, the main idea behind this project is building a model with historical data from customers, and assign a score of the likelihood of churning. 

For this project, we used a [Kaggle dataset](https://www.kaggle.com/blastchar/telco-customer-churn). 


## 3.2 Data Preparation

This session covered data obtention, and some procedures of data preparation. 

**Commands, functions and methods:** 

* `!wget` - Linux shell command for downloading data 
* `pd.read.csv()` - read csv files 
* `df.head()` - take a look of the dataframe 
* `df.head().T` - take a look of the transposed dataframe 
* `df.columns` - retrieve colum names of a dataframe 
* `df.columns.str.lower()` - lowercase all the letters 
* `df.columns.str.replace(' ', '_')` - replace the space separator 
* `df.dtypes` - retrieve data types of all series 
* `df.index` - retrive indices of a dataframe
* `pd.to_numeric()` - convert a series values to numerical values. The `errors=coerce` argument allows to make the transformation despite of some encountered errors. 
* `df.fillna()` - replace NAs with some value 
* `(df.x == "yes").astype(int)` - convert x series of yes-no values to numerical values. 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb).  

## 3.3 Setting Up The Validation Framework

Splitting the dataset with **Scikit-Learn**. 

**Classes, functions and methods:** 

* `train_test_split` - Scikit-Learn class for splitting datasets. Linux shell command for downloading data. The `random_state` argument set a random seed for reproducibility purpouses.  
* `df.reset_index(drop=True)` - reset the indices of a dataframe and delete the previous ones. 
* `df.x.values` - extract the values from x series
* `del df['x']` - delete x series from a dataframe 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb).  

## 3.4 EDA

The EDA for this projects consisted of: 
* Checking missing values 
* Looking at distribution of the target variable (churn)
* Looking at numerical and categorical variables 

**Functions and methods:** 

* `df.isnull().sum()` - retunrs the number of null values in the dataframe.  
* `df.x.value_counts()` returns the number of values for each category in x series. The `normalize=True` argument retrieves the percentage of each category. In this project, the mean of churn is equal to the churn rate obtained with value_counts method. 
* `round(x, y)` - round a x number with y decimal places
* `df[x].nunique()` - returns the number of unique values in x series 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb). 


## 3.5 Feature Importance: Churn Rate And Risk Ratio

1. **Churn rate:** Difference between mean of the target variable and mean of categories for a feature. If this difference is greater than 0, it means that the category is less likely to churn, and if the difference is lower than 0, the group is more likely to churn. The larger differences are indicators that a variable is more important than others. 

2. **Risk ratio:** Ratio between mean of categories for a feature and mean of the target variable. If this ratio is greater than 1, the category is more likely to churn, and if the ratio is lower than 1, the category is les lekely to churn. It expresses the feature importance in relative terms. 

**Functions and methods:** 

* `df.groupby('x').y.agg([mean()])` - returns a datframe with mean of y series grouped by x series 
* `display(x)` displays an output in the cell of a jupyter notebook. 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb). 

## 3.6 Feature Importance: Mutual Information

Mutual information is a concept from information theory, which measures how much we can learn about one variable if we know the value of another. In this project, we can think of this as how much do we learn about churn if we have the information from a particular feature. So, it is a measure of the importance of a categorical variable. 

**Classes, functions and methods:** 

* `mutual_info_score(x, y)` - Scikit-Learn class for calculating the mutual information between the x arget variable and the y feature. 
* `df[x].apply(y)` - apply a y function to the x series of the df dataframe. 
* ` df.sort_values(ascending=False).to_frame(name='x')` - sort values in an ascending order and called the column as x. 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb). 

## 3.7 Feature Importance: Correlation

**Correlation coefficient** measures the degree of dependency between two variables. This value is negative if one variable growths while the other decreases, and it is positive if both variables increase. Depending on its size, the dependency between both varibales could be low, moderate, or strong. It allows to measure the importance of numerical variables. 

**Functions and methods:** 

* `df[x].corrwith(y)` - returns the correlation between x and y series. 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb). 

## 3.8 One-Hot Encoding

One-Hot Encoding allows to encode categorical variables in numerical ones. This method represent each category of a variable as one column, and a 1 is assigned if the value belongs to the category or 0 otherwise. 

**Classes, functions and methods:** 

* `df[x].to_dict(oriented='records')` - convert x series to dictionaries, oriented by rows. 
* `DictVectorizer().fit_transform(x)` - Scikit-Learn class for converting x dictionaries into a sparse matrix, and in this way doing the one hot encoding. It does not affect the numerical variables. 
* `DictVectorizer().get_feature_names()` -  returns the names of the columns in the sparse matrix.  

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb). 

## 3.9 Logistic Regression

In general, supervised models follow can be represensted with this formula: 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large g\left(x_{i}\right) = y_{i}"/>
</p>

Depending on what is the type of target variable, the supervised task can be regression or classification (binary or multiclass). Binary classification tasks can have negative (0) or positive (1) target values. The output of these models is the probability of xi belonging to the positive class.  

Logistic regression is similar to linear regression because both models take into account the bias term and weighted sum of features. The difference between these models is that the output of linear regression is a real number, while logistic regression outputs a value between zero and one, applying the sigmoid function to the linear regression formula. 

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large g\left(x_{i}\right) = Sigmoid\left(w_{0} %2B w_{1}x_{1} %2B w_{2}x_{2} %2B ... %2B w_{n}x_{n}\right)"/>
</p>

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\large Sigmoid=\frac{1}{1%2Bexp\left( -z \right)}"/>
</p>

In this way, sigmoid function allows to transform a score into a probability. 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb). 

## 3.10 Training Logistic Regression with Scikit-Learn

This video was about training logistic regression model with Scikit-Learn, applying it to the validation dataset, and calculating its accuracy. 

**Classes, functions and methods:** 

* `LogisticRegression().fit_transform(x)` - Scikit-Learn class for calculating the logistic regression model. 
* `LogisticRegression().coef_[0]` - returns the coeffcients or weights of the LR model
* `LogisticRegression().intercept_[0]` - returns the bias or intercept of the LR model
* `LogisticRegression().predict[x]` - make predictions on the x dataset 
* `LogisticRegression().predict_proba[x]` - make predictions on the x dataset, and returns two columns with their probabilities for the two categories - soft predictions 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb). 


## 3.11 Model Interpretation

This video was about the interpretation of coefficients, and training a model wit fewer featutes. 

In the formula og logistic regression model, only one of the the one hot encoded categories is multiplied by 1, and the other by 0. In this way, we only consider the appropriate category for each categorical feature. 

**Classes, functions and methods:** 

* `zip(x,y)` - returns a new list with elements from x joined with their corresponding elements on y 

The entire code of this project is available at [this jupyter notebook](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/03-churn.ipynb). 


## 3.12 Using the model 

We trained the logistic regression model with the full training dataset (training + validation), considering numerical and categorical features. Thus, predictions were made on the test dataset, and we evaluate the model using the accuracy metric. 

In this case, the predictions of validation and test were similar, which means that the model is working well. 

## 3.13 Summary

In this session, we worked on a project to predict churning in customers from a company. We learned feature importance of numerical and categorical variables, including risk ratio, mutual information and correlation coefficient. Also, we understood one-hot encoding, and implemented logistic regression with Scikit-Learn. 