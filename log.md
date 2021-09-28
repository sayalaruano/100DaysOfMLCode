# 100 Days Of ML Code - Log

Hi! I am Sebastián, a Machine Learning enthusiast and this is my log for the 100DaysOfMLCode Challenge.

## Day 1: September 10, 2021

**Today's Progress:** I have completed the [first week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/01-intro). 
In general, I learnt about concept of ML with an example of prediction of car's prizes, differences between ML vs Rule-bases systems, supervised ML tasks, the 
CRISP-DM methodology for organizing ML projects, model selection process, and a quick recap of numpy, linear algebra and pandas. 

**Thoughts:** I liked how the general concepts of ML were presented in the mlzoomcamp, which had examples for each section, and it clarified all the contents. 
I already knew most of the content from recap of numpy, linear algebra and pandas, but it was a nice summary. I enjoyed with the homework, although the 
exercises were not very difficult. However, this review was useful to remember all the concepts that I will use later in the course and other ML projects. 

**Link of Work:** 
* [NotesDay1](Notes/NotesDay1.md)
* [Jupyter notebook of the homework for the first week of mlzoomcamp](Intro_ML/Homework_week1_mlzoomcamp.ipynb)

## Day 2: September 11, 2021

**Today's Progress:** I have completed the first chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). 
In general, I learnt about concept the of ML, why these models are important and some examples, types of ML systems, the main challenges of ML, the importance of 
splitting a ML model in training, validation and test datasets, and other relevant general concepts. 

**Thoughts:** It was interesting the list of main challenges of ML models and some strategies for avoiding them. Also, I realized that the best model obtained after evaluating the validation set, is trained 
with train+val datasets, and what is the importance of cross-validation strategy. Another interesting idea was the *No Free Lunch Theorem* because it reflect that we need to make 
assumptions about the data to choose a few reasonable models, instead of testing all of them. Something that I did not understant was the **train-dev set**. 

**Link of Work:** 
* [NotesDay2](Notes/NotesDay1.md)

## Day 3: September 12, 2021

**Today's Progress:** I have completed the second chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). 
In general, I learnt about the steps followed to make a complete ML project, including understanding of the problem, data obtention, exploratory data analysis, 
data cleaning and pre-processing, feature engineering, model's training and evaluation, cross-validation, fine-tuning of model's hyperparameters, model deployment. 

**Thoughts:** This chapter was full of a lot of useful concepts and advices. Some interesting ideas an concepts were: l1 and l2 norms, sklearn classes for data 
pre-processing and pipelines, general sklearn design of classes, cross-validation, GridSearchCV and RandomizedSearchCV, and the code examples to perform all the analysis. 

**Link of Work:** 
* [NotesDay3](Notes/NotesDay3.md)

## Day 4: September 13, 2021

**Today's Progress:** I watched the [first part of the lecture Traditional Feature-based Methods](https://www.youtube.com/watch?v=3IS7UhNMQ3U&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=4) 
from the Stanford's course [Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/). This lecture covered the graph's features that can be used for 
training ML models, particularly those derived from nodes. The information-based features included node degree and some centrality measures such as eigenvector, 
betweenness, and closeness centralities. The structure-based features were node degree, clustering cofficient, and graphlet degree vectors. All f these features can 
be used for making predictions of unknown labels from certain nodes. 

**Thoughts:** The lecture had a lot of useful concepts of graph features used for training ML models, particularly the node features. I enjoyed reminding some concepts 
of network science such as centrality measures, clustering coefficient, and graphlets. However, I am intrigued about the way by which all of these features are 
converted to data that the ML model can interpret. 

**Link of Work:** 
* [NotesDay4](Notes/NotesDay4.md)

## Day 5: September 14, 2021

**Today's Progress:** I have completed half of the content for the [second week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/02-regression). 
In general, I learnt about data preparation, exploratory data analysis, setting up the validation framework, and the application of linear regression model for predicting car prices. Also, we 
understand the internals of linear regression. 

**Thoughts:** I enjoyed the videos of this session, especially the understanding of linear regression model in its vectorized form, and how it can be solved by 
finding the vector of weights or coefficients form the Normal equation. 

**Link of Work:** 
* [NotesDay5](Notes/NotesDay5.md)

## Day 6: September 15, 2021

**Today's Progress:** I have completed all the content for the [second week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/02-regression). 
This material included data preparation, exploratory data analysis, the validation framework, linear regression model, LR vector and normal forms, the baseline model, root mean squared error, feature engineering, regularization, tuning the model, 
and using the best model with new data. All these concepts were explanied using the problem to preditc the price of cars. 

**Thoughts:** For the first time, I understand regularization and what it represents in the feature matrix, which in brief adds some small values to the 
diagonal of this matrix, and in this way there are not duplicated colummns. Also, it was interesting to see the entire workflow in a regression problem, including
explanations of all of these secions. 

**Link of Work:** 
* [NotesDay6](Notes/NotesDay6.md)

## Day 7: September 16, 2021

**Today's Progress:** I have completed the homework for the [second week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/02-regression/homework.md). 
The homework was about the creation of a regression model for predicting apartment prices using Kaggle's dataset  [New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv). 

**Thoughts:** I enjoyed this homework because it synthesized all the contents learned during the second week of mlzoomcamp. Also, it was great to work with real 
data and apply all the steps of pre-processing, data exploratory analysis, regularization, and fine-tuning. 

**Link of Work:** 
* [Jupyter notebook of the homework for the second week of mlzoomcamp](Regression/Homework_week2_mlzoomcamp.ipynb)

## Day 8: September 17, 2021

**Today's Progress:** I built a regression model for the Kaggle's competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
The competition was about the creation of a regression model for predicting house prices using the dataset of this competition. 
I applied some new tools for me, including sklearn and pandas profiling libraries. 

**Thoughts:** This competition was nice because I practiced all what I learned about regression on a real problem. Also, I explored new libraries such as
pandas profiling for EDA and Scikit-learn for pre-processing, regularization, fine-tuning, and training my models. 

**Link of Work:** 
* [Jupyter notebook of the House Prices - Advanced Regression Techniques competition](Regression/House_Prices_Advanced_Regression_Techniques_Kaggle.ipynb)

## Day 9: September 20, 2021

**Today's Progress:** I attended the [week 3 ML Zoomcamp Office Hours](https://www.youtube.com/watch?v=JC3TQw34-m0). In this session, we reviewed the homework of this week and answered questions regarding regression problems. 

**Thoughts:** I verified that all answers of my homework were correct, and I also learned different ways to solve the same questions. 

**Link of Work:** 
* [Jupyter notebook of the homework for the second week of mlzoomcamp](Regression/Homework_week2_mlzoomcamp.ipynb)

## Day 10: September 21, 2021

**Today's Progress:** I read most of the fourth chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). 
I learnt about Linear regression, and different ways to train this model, including the Normal Equation, Singular Value Descomposition, and different implementations of Gradient descent (Batch, Stochastic, and mini-batch). Also, this chapter had information about polynomial regression, learning curves, and regularized linear models, including Ridge, Lassso, and ElasticNet. 

**Thoughts:** I enjoyed to learn different ways for training a linear model, as well as their advantages and downsides. Also, I understand how to interpret the learning curves, and the errors associated to underfitting and overfitting models. Finally, it was interesting to learn different ways to regularize a linear regression model. 

**Link of Work:** 
* [NotesDay10](Notes/NotesDay10.md)

## Day 11: September 22, 2021

**Today's Progress:** I studied all contents for the [thrid week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/03-classification). In this session, we worked on a project to predict churning in customers from a company. We learned feature importance of numerical and categorical variables, including risk ratio, mutual information and correlation coefficient. Also, we understood one-hot encoding, and implemented logistic regression with Scikit-Learn.  

**Thoughts:** I learned some useful concepts that I have never undertood, including risk ratio and mutual informations as metrics to measure the feature importance of categorical variables. Also, it was interesting to learn about logistic regression and sigmoid function as a model for solving a binary classification task. 

**Link of Work:** 
* [NotesDay11](Notes/NotesDay11.md)

## Day 12: September 24, 2021

**Today's Progress:** I completed the homework for the [third week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/03-classification/homework.md). 
The homework was about the creation of a classification model for predicting apartment prices using Kaggle's dataset  [New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv). We used Scikit-Learn classes and methods for all stages of this project. 

**Thoughts:** This homework was nice because it summarized all contents of the third week of mlzoomcamp. I encountered some problems with the convergence of Scikit-Learn's Logistic regression with the method `lbfgs`. Also, I realized that for classification tasks all the target values must be binarized to make the predictions. 

**Link of Work:** 
* [Jupyter notebook of the homework for the third week of mlzoomcamp](Classification/Homework_week3_mlzoomcamp_classification.ipynb)

## Day 13: September 27, 2021

**Today's Progress:** I attended the [week 4 ML Zoomcamp Office Hours](https://www.youtube.com/watch?v=JC3TQw34-m0). In this session, we reviewed the homework of this week and answered questions regarding logistic regression and classification tasks. Also, I read the Logistic regression content from the fourth chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).  

**Thoughts:** I verified that all answers of my homework were correct, and learned different ways to solve these questions. Also, I learned about Softmax Regression as a generalization of Logistic regression, and its cost function - cross entropy.

**Link of Work:** 
* [Jupyter notebook of the homework for the third week of mlzoomcamp](Classification/Homework_week3_mlzoomcamp_classification.ipynb)
* [NotesDay13](Notes/NotesDay11.md)