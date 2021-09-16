# NotesDay5 - Machine Learning Zoomcamp second week 

## 2.8 Baseline model for car price prediction project

The LR model obtained in the previous section was used with the dataset of car price prediction. For this model, only the numerical variables were considered. 
The training data was pre-processed, replacing the NaN values with 0, in such a way that these values were omitted by the model. Then, the model was trained 
and it allowed to make predictions on new data. Finally, distributions of y target variable and predictions were compared by plotting their histograms. 


## 2.9 Root mean squared error (RMSE)

The RMSE is a measure of the error associated to a model for regression tasks. The video explained the RMSE formula in detail and implemented it in Python. 


## 2.10 Computing RMSE on Validation Data

Calculation of the RMSE on validation partition of the dataset of car price prediction. In this way, we have a metric to evaluate the model's 
performance. 


## 2.11 Feature Engineering

The feature age of the car was included in the dataset, obtained with the substraction of the maximum year of cars and each of the years of cars. 
This new feature improved the model performance, measured with the RMSE and comparing the distributions of y target variable and predictions. 

## 2.12 Categorical Variables

Categorical variables are typically strings, and pandas identify them as object types. These variables need to be converted to a numerical form because the ML
models can interpret only numerical features. It is possible to incorporate certain categories from a feature, not necesary all of them. 
This transformation from categorical to numerical variables is known as One-Hot encoding. 

## 2.13 Regularization 

If the feature matrix has duplicated columns, it does not have an inverse matrix. But, sometimes this error could be passed if certain values are slightely different
between duplicated columns. So, if we apply the normal equation with this feature matrix, the values associated to duplicated columns are very large, which decreases
the model performance. To solve this issue, one alternative is adding a small number to the diagonal of feature matrix, which corresponds to regularization. This technique 
works because the addition of small values to the diagonal makes less likely to have duplicated columns. The regularization value is a parameter of the model. After applying 
regularization the model performance improved. 

## 2.14 Tuning the model 

The tuning consisted of finding the best regularization value, using validation partition of the dataset. After obtaining the best regularization value, the model 
was trained with this regularization parameter. 

## 2.15 Using the model 

After finding the best model and its parameters, it was trained with training and validation partitions and the final evaluation was calculated on the test partition. 
Finally, the final model was used t predict the price of a new cars. 

## 2.16 Next steps 

In summary, this session covered some topics, including data preparation, exploratory data analysis, the validation framework, linear regression model, LR vector and 
normal forms, the baseline model, root mean squared error, feature engineering, regularization, tuning the model, and using the best model with new data. All these concepts 
were explanied using the problem to preditc the price of cars. 




