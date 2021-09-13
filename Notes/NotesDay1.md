# NotesDay1

## Machine Learning Zoomcamp first week 
### 1. Introduction to ML
The concept of ML was depicted with an example of predicting the price of a car. The ML model learns from data, represented as 
some **features** such as year, mileage, among others, and the **target** variable, in this case, the car's prize. Then, the model extracts patterns 
from the data and predicts the price of cars that were not considered in the training data. 

In summary, ML is a process of **extracting patterns from data**, which is of two types: features (information about the object) and target (correct predictions). 
Therefore, new features are presented to the model, and it makes **predictions** considering learned information. 

## 2. ML vs Rule-Based Systems
The differences between ML and Rule-Based systems were explained with the example of a **spam filter**. The traditional systems are based on a set of 
characteristics that identify an email as spam, which have some drawbacks because the spam emails keep changing over time and the system must be upgraded with 
these adjustments, and this process is untractable due to code maintenance and other issues. The ML systems can be trained with features and targets extracted 
from the rules of Rule-Based systems, a model is obtained, and it can be used to predict new emails as spam or normal emails. The **predictions are probabilities**, and to make a decision it is necessary to define a threshold to classify emails as spam or normal ones.  

## 3. Supervised machine learning 
In SML there are always labels associated with certain features. The model is trained, and then it can make predictions on new features. In this way, the model
is taught by certain features and targets. 

**Feature matrix (X):** made of observations (rows) and features (columns). For each row, there is a **y** associated vector of targets. 
The model can be represented as a function **g** that takes the X matrix as a parameter and tries to predict values as close as possible to y targets. 
The obtention of the g function is what it is called **training**.

### Types of SML problems 
* Regression - the output is a number (car's prize)
* Classification - the output is a category (spam example). 
	* Binary - there are two categories. 
	* Multiclass problems - there are more than two categories. 
* Ranking - the output is the big scores associated with certain items. It is applied in recommender systems. 

In summary, SML is about teaching the model by showing different examples, and the goal is to come up with a function that takes the feature matrix as a
parameter and makes predictions as close as possible to the y targets. 

## 4. CRISP-DM ML process 
CRISP-DM is a methodology for organizing ML projects. It was invented in the 90s by IBM. The steps of this procedure are: 
1. Business understanding: An important question is if do we need ML for the project. The goal of the project has to be measurable. 
2. Data understanding: Analyze available data sources, and decide if more data is required. 
3. Data preparation: Clean data and remove noise applying pipelines, and the data should be converted to a tabular format, so we can put it into ML.
4. Modeling: training Different models and choose the best one. Considering the results of this step, it is proper to decide if is required to add new features or fix data issues. 
5. Evaluation: Measure how well the model is performing and if it solves the business problem. 
6. Deployment: Roll out to production to all the users. The evaluation and deployment often happen together - **online evaluation**. 
It is important to consider how well maintainable the project is.
  
In general, ML projects require many iterations. 

## 5. Model Selection Process

The validation dataset is not used in training. There are feature matrices and y vectors for both training and validation datasets. 
The model is fitted with training data, and it is used to predict y values of the validation feature matrix. Then, the predicted the y values (probabilities)
are compared with the actual y values. 

**Multiple comparisons problem:** just by chance one model can be lucky and obtain good predictions because all of them are probabilistic. 

The test set can help to avoid the MCP. Obtention of the best model is done with the training and validation datasets, while the test dataset is used for assuring that the proposed best model is the best. 

1. Split datasets in training, validation, and test. 
2. Train the models
3. Evaluate the models
4. Select the best model 
5. Apply the best model to the test dataset 
6. Compare the performance metrics of validation and test 






