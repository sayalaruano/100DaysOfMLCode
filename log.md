# 100 Days Of ML Code - Log

Hi! I am Sebastián, a Machine Learning enthusiast and this is my log for the 100DaysOfMLCode Challenge.

## Day 1: September 10, 2021

**Today's Progress:** I have completed the [first week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/01-intro).
In general, I learnt about concept of ML with an example of prediction of car's prizes, differences between ML vs Rule-bases systems, supervised ML tasks, the CRISP-DM methodology for organizing ML projects, model selection process, and a quick recap of numpy, linear algebra and pandas.

**Thoughts:** I liked how the general concepts of ML were presented in the mlzoomcamp, which had examples for each section, and it clarified all the contents.
I already knew most of the content from recap of numpy, linear algebra and pandas, but it was a nice summary. I enjoyed with the homework, although the exercises were not very difficult. However, this review was useful to remember all the concepts that I will use later in the course and other ML projects.

**Link of Work:**

* [NotesDay1](Intro_ML/Notes/NotesDay1.md)
* [Jupyter notebook of the homework for the first week of mlzoomcamp](Intro_ML/Notebooks/Homework_week1_mlzoomcamp_car_price_prediction.ipynb)

## Day 2: September 11, 2021

**Today's Progress:** I have completed the first chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).
In general, I learnt about concept the of ML, why these models are important and some examples, types of ML systems, the main challenges of ML, the importance of splitting a ML model in training, validation and test datasets, and other relevant general concepts.

**Thoughts:** It was interesting the list of main challenges of ML models and some strategies for avoiding them. Also, I realized that the best model obtained after evaluating the validation set, is trained with train+val datasets, and what is the importance of cross-validation strategy. Another interesting idea was the *No Free Lunch Theorem* because it reflect that we need to make assumptions about the data to choose a few reasonable models, instead of testing all of them. Something that I did not understant was the **train-dev set**.

**Link of Work:**

* [NotesDay2](Intro_ML/Notes/NotesDay1.md)

## Day 3: September 12, 2021

**Today's Progress:** I have completed the second chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).
In general, I learnt about the steps followed to make a complete ML project, including understanding of the problem, data obtention, exploratory data analysis, data cleaning and pre-processing, feature engineering, model's training and evaluation, cross-validation, fine-tuning of model's hyperparameters, model deployment.

**Thoughts:** This chapter was full of a lot of useful concepts and advices. Some interesting ideas an concepts were: l1 and l2 norms, sklearn classes for data pre-processing and pipelines, general sklearn design of classes, cross-validation, GridSearchCV and RandomizedSearchCV, and the code examples to perform all the analysis.

**Link of Work:**

* [NotesDay3](Intro_ML/Notes/NotesDay3.md)

## Day 4: September 13, 2021

**Today's Progress:** I watched the [first part of the lecture Traditional Feature-based Methods](https://www.youtube.com/watch?v=3IS7UhNMQ3U&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=4) from the Stanford's course [Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/). This lecture covered the graph's features that can be used for training ML models, particularly those derived from nodes. The information-based features included node degree and some centrality measures such as eigenvector, betweenness, and closeness centralities. The structure-based features were node degree, clustering cofficient, and graphlet degree vectors. All f these features can be used for making predictions of unknown labels from certain nodes.

**Thoughts:** The lecture had a lot of useful concepts of graph features used for training ML models, particularly the node features. I enjoyed reminding some concepts of network science such as centrality measures, clustering coefficient, and graphlets. However, I am intrigued about the way by which all of these features are converted to data that the ML model can interpret.

**Link of Work:**

* [NotesDay4](Graph_ML/Notes/NotesDay4.md)

## Day 5: September 14, 2021

**Today's Progress:** I have completed half of the content for the [second week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/02-regression).
In general, I learnt about data preparation, exploratory data analysis, setting up the validation framework, and the application of linear regression model for predicting car prices. Also, we understand the internals of linear regression.

**Thoughts:** I enjoyed the videos of this session, especially the understanding of linear regression model in its vectorized form, and how it can be solved by finding the vector of weights or coefficients form the Normal equation.

**Link of Work:**

* [NotesDay5](Regression/Notes/NotesDay5.md)

## Day 6: September 15, 2021

**Today's Progress:** I have completed all the content for the [second week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/02-regression). This material included data preparation, exploratory data analysis, the validation framework, linear regression model, LR vector and normal forms, the baseline model, root mean squared error, feature engineering, regularization, tuning the model, and using the best model with new data. All these concepts were explanied using the problem to predict the price of cars.

**Thoughts:** For the first time, I understand regularization and what it represents in the feature matrix, which in brief adds some small values to the diagonal of this matrix, and in this way there are not duplicated columns. Also, it was interesting to see the entire workflow in a regression problem, including explanations of all of these sections.

**Link of Work:**

* [NotesDay6](Regression/Notes/NotesDay6.md)

## Day 7: September 16, 2021

**Today's Progress:** I have completed the homework for the [second week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/02-regression/homework.md). The homework was about the creation of a regression model for predicting apartment prices using Kaggle's dataset  [New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv).

**Thoughts:** I enjoyed this homework because it synthesized all the contents learned during the second week of mlzoomcamp. Also, it was great to work with real data and apply all the steps of pre-processing, data exploratory analysis, regularization, and fine-tuning.

**Link of Work:**

* [Jupyter notebook of the homework for the second week of mlzoomcamp](Regression/Notebooks/Homework_week2_mlzoomcamp_reg_car_price_pred.ipynb)

## Day 8: September 17, 2021

**Today's Progress:** I built a regression model for the Kaggle's competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
The competition was about the creation of a regression model for predicting house prices using the dataset of this competition. I applied some new tools for me, including sklearn and pandas profiling libraries.

**Thoughts:** This competition was nice because I practiced all what I learned about regression on a real problem. Also, I explored new libraries such as
pandas profiling for EDA and Scikit-learn for pre-processing, regularization, fine-tuning, and training my models.

**Link of Work:**

* [Jupyter notebook of the House Prices - Advanced Regression Techniques competition](Regression/Notebooks/House_Prices_Advanced_Regression_Techniques_Kaggle.ipynb)

## Day 9: September 20, 2021

**Today's Progress:** I attended the [week 3 ML Zoomcamp Office Hours](https://www.youtube.com/watch?v=JC3TQw34-m0). In this session, we reviewed the homework of this week and answered questions regarding regression problems.

**Thoughts:** I verified that all answers of my homework were correct, and I also learned different ways to solve the same questions.

**Link of Work:**

* [Jupyter notebook of the homework for the second week of mlzoomcamp](Regression/Notebooks/Homework_week2_mlzoomcamp_reg_car_price_pred.ipynb)

## Day 10: September 21, 2021

**Today's Progress:** I read most of the fourth chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). I learnt about Linear regression, and different ways to train this model, including the Normal Equation, Singular Value Descomposition, and different implementations of Gradient descent (Batch, Stochastic, and mini-batch). Also, this chapter had information about polynomial regression, learning curves, and regularized linear models, including Ridge, Lassso, and ElasticNet.

**Thoughts:** I enjoyed to learn different ways for training a linear model, as well as their advantages and downsides. Also, I understand how to interpret the learning curves, and the errors associated to underfitting and overfitting models. Finally, it was interesting to learn different ways to regularize a linear regression model.

**Link of Work:**

* [NotesDay10](Regression/Notes/NotesDay10.md)

## Day 11: September 22, 2021

**Today's Progress:** I studied all contents for the [thrid week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/03-classification). In this session, we worked on a project to predict churning in customers from a company. We learned feature importance of numerical and categorical variables, including risk ratio, mutual information and correlation coefficient. Also, we understood one-hot encoding, and implemented logistic regression with Scikit-Learn.  

**Thoughts:** I learned some useful concepts that I have never understood, including risk ratio and mutual information as metrics to measure the feature importance of categorical variables. Also, it was interesting to learn about logistic regression and sigmoid function as a model for solving a binary classification task.

**Link of Work:**

* [NotesDay11](Classification/Notes/NotesDay11.md)

## Day 12: September 24, 2021

**Today's Progress:** I completed the homework for the [third week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/03-classification/homework.md). The homework was about the creation of a classification model for predicting apartment prices using Kaggle's dataset  [New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv). We used Scikit-Learn classes and methods for all stages of this project.

**Thoughts:** This homework was nice because it summarized all contents of the third week of mlzoomcamp. I encountered some problems with the convergence of Scikit-Learn's Logistic regression with the method `lbfgs`. Also, I realized that for classification tasks all the target values must be binarized to make the predictions.

**Link of Work:**

* [Jupyter notebook of the homework for the third week of mlzoomcamp](Classification/Notebooks/Homework_week3_mlzoomcamp_classification_churning.ipynb)

## Day 13: September 27, 2021

**Today's Progress:** I attended the [week 4 ML Zoomcamp Office Hours](https://www.youtube.com/watch?v=JC3TQw34-m0). In this session, we reviewed the homework of this week and answered questions regarding logistic regression and classification tasks. Also, I read the Logistic regression content from the fourth chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).  

**Thoughts:** I verified that all answers of my homework were correct, and learned different ways to solve these questions. Also, I learned about Softmax Regression as a generalization of Logistic regression, and its cost function - cross entropy.

**Link of Work:**

* [Jupyter notebook of the homework for the third week of mlzoomcamp](Classification/Notebooks/Homework_week3_mlzoomcamp_classification_churning.ipynb)
* [NotesDay13](Classification/Notes/NotesDay11.md)

## Day 14: September 29, 2021

**Today's Progress:** I have completed half of the content for the [fourth week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/02-regression). In general, I learnt about different metrics to evaluate a binary classifier, including accuracy, confusion table, precision and recall.  

**Thoughts:** I enjoyed the videos of this session, especially the understanding of confusion table, a metric that I have never seen before.

**Link of Work:**

* [NotesDay14](Classification/Notes/NotesDay14.md)

## Day 15: September 30, 2021

**Today's Progress:** I have completed all the content for the [fourth week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/02-regression). I learnt about different metrics to evaluate a binary classifier, including ROC curves and AUROC. Also, I understand the concept of cross-validation to evaluate the model in different partitions and obtain a more robust result.

**Thoughts:** I really like to understand the concepts of ROC curves and AUROC in a extended manner, which I have never did before.

**Link of Work:**

* [NotesDay15](Classification/Notes/NotesDay15.md)

## Day 16: October 1, 2021

**Today's Progress:** I have completed the homework for the [fourth week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/04-evaluation/homework.md). The homework was about the evaluation of a classifier using different metrics learned during this lesson. The dataset was about  credit scoring, and it is available at [this link](https://github.com/gastonstat/CreditScoring).

**Thoughts:** I enjoyed this homework because it synthesized all the contents learned during the fourth week of mlzoomcamp. It was interesting to understand the PR curves, F1 score, and cross-validation.

**Link of Work:**

* [Jupyter notebook of the homework for the fourth week of mlzoomcamp](Classification/Homework_week4_mlzoomcamp_classification_credit_scoring.ipynb)

## Day 17: October 3, 2021

**Today's Progress:** I read the third chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). I learnt about binary classifiers, cross-validation, metrics to evaluate these models such as accuracy, confusion matrix, precision, recall, F1 score, ROC curves, and AUROC. Also, I understood the differences between multiclass, multilabel, and multioutput classifiers.

**Thoughts:** I found out interesting stratified k-fold cross-validation, precision/recall trade-off, PR and ROC curves, OvR and OvO strategies for multiclass classifiers, and multi-output classifiers.

**Link of Work:**

* [NotesDay17](Classification/Notes/NotesDay17.md)

## Day 18: October 4, 2021

**Today's Progress:** I attended the [week 5 ML Zoomcamp Office Hours](https://www.youtube.com/watch?v=JC3TQw34-m0). In this session, we reviewed the homework of this week and answered questions regarding metrics to evaluate a binary classifier such as confusion matrix, accuracy, precision, recall, F1 score, and AUROC.

**Thoughts:** I verified that all answers of my homework were correct, and learned different ways to solve these questions.

**Link of Work:**

* [Jupyter notebook of the homework for the fourth week of mlzoomcamp](Classification/Homework_week4_mlzoomcamp_classification_credit_scoring.ipynb)

## Day 19: October 7, 2021

**Today's Progress:** I have completed half of the content for the [fifth week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment).
In general, I learnt about how to save and load a machine learning model, an introduction to flask for creating a web service, and the process for serving a machine learning model as a web service.
**Thoughts:** I loved to learn about flask, web services, and how to deploy a machine learning model in this platform. All of these knowledge was new for me, and I really enjoyed to learn about it.

**Link of Work:**

* [NotesDay19](Deployment/Notes/NotesDay19.md)

## Day 20: October 7, 2021

**Today's Progress:** I have studied all the content for the [fifth week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment).
In general, I learnt about python virtual environment management using pienv, system Environment management with Docker, and how to deploy the churning model to the cloud with AWS Elastic Beanstalk.
**Thoughts:** I enjoyed to learn about managements of python and systems environments. This was a topic I wanted to learn some time ago, and the contents of the MLZoomcamp were incredible. Now, I want to learn how to use free cloud services for deploying my ML models.

**Link of Work:**

* [NotesDay20](Deployment/Notes/NotesDay20.md)

## Day 21: October 8, 2021

**Today's Progress:** I completed the homework for the [fifth week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment/homework.md). The homework was about loading a churning prediction ML model with pickle, serving it as a web service with flask, and using docker to isolate the environment for its execution.

**Thoughts:** I loved learning useful tools for the deployment of ML models, including pickle, flask, pipenv, docker, among others. All about this topic was new for me, and I enjoyed a lot by learning this.

**Link of Work:**

* [md file with answers and links to python and docker file](Deployment/Homework_ml-zoomcamp_fifth_week/Homework_ml-zoomcamp_fifth_week.md)

## Day 22: October 11, 2021

**Today's Progress:** I attended the [week 6 ML Zoomcamp Office Hours](https://www.youtube.com/watch?v=JC3TQw34-m0). In this session, we reviewed the homework of the 5th week and answered questions regarding web services with pickle, pipenv, flask, gunicorn, docker, among other tools.

**Thoughts:** I was wrong in two questions of the homework. These questions were about docker, so I need to review this content of the course.

**Link of Work:**

* [Answers of the homework for the fifth week of mlzoomcamp](Deployment/Homework_ml-zoomcamp_fifth_week.md)

## Day 23: October 14, 2021

**Today's Progress:** I have completed half of the content for the [sixth week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees).
In general, I understand credit risk scoring project, its data preparation, the concept of decision trees and how it is the learning process of this algorithm.
**Thoughts:** I liked to learn in deep the details behind decision trees and its learning process.

**Link of Work:**

* [NotesDay23](Decision_trees_Ensemble_learning/Notes/NotesDay23.md)

## Day 24: October 17, 2021

**Today's Progress:** I have completed all content for the [sixth week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees).
In general, I learned about ensemble models, random forest, gradient boosting trees, and how to fine-tune these models.
**Thoughts:** It was interesting to learn differences between random forest and gradient boosting trees, two types of ensemble models. Also, I liked to know about important parameters for these models and how to fine-tune them. 

**Link of Work:**

* [NotesDay24](Decision_trees_Ensemble_learning/Notes/NotesDay24.md)

## Day 25: October 19, 2021

**Today's Progress:** I attended the [week 7 ML Zoomcamp Office Hours](https://youtu.be/wWBm6MHu5u8). In this session, we reviewed information about the midterm project.

**Thoughts:** I decided to work in the Open Bioinformatics Research Project proposed by Data Professor. This project is related to drug discovery studying betalactamase proteins. The dataset of this project is available [here](https://kaggle.com/thedataprof/betalactamase) and detailed information about project is explained in [this video](https://youtu.be/_GtEgiWWyK4) of Data Professor channel.

## Day 26: October 20, 2021

**Today's Progress:** I completed the homework for the [sixth week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/06-trees/homework.md). In this homework, we created tree based models for solving a regression task, which was predicting apartment prices using Kaggle's dataset  [New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv).

**Thoughts:** I enjoyed this homework because I understand properly how tree based models work, and we applied these models for solving a regression task.

**Link of Work:**

* [Jupyter notebook of the homework for the sixth week of mlzoomcamp](Decision_trees_Ensemble_learning/Notebooks/Homework_week6_mlzoomcamp_tree_based_reg_airbnb_dataset.ipynb)

## Day 27: October 24, 2021

**Today's Progress:** I started my midterm project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project).  I decided to work in the Open Bioinformatics Research Project proposed by [Data Professor](https://github.com/dataprofessor), which is related to Computational Drug Discovery. I did the README of my project, a preliminary exploratory data analysis, and the calculation of 12 fingerprints for all molecules. 

**Thoughts:** I enjoyed doing my project because drug discovery is the field that capture my interest for postgraduate studies, and it was really nice to learn more about it. I liked to learn about basic concepts of drug discovery, and how to apply this knowledge in a machine learning project.

**Link of Work:**

* [GitHub repository of my midterm project for the sixth week of mlzoomcamp](https://github.com/sayalaruano/MidtermProject-MLZoomCamp)

## Day 29: October 26, 2021

**Today's Progress:** I continued with my midterm project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I used Logistic Regression and Random Forest to train classifiers for my project. For this, I learned how to work with GridSearchCV class of sklearn, which allows to perform feature tuning and cross validation of sklearn models.

**Thoughts:** I learned a lot working with GridSearchCV and tuning my classifiers to train the data of my project. However, the performance metrics of models I have tried have been very low, and I think that it can e related to overfitting. So, I need to come up with solutions to improve my classifier.

**Link of Work:**

* [GitHub repository of my midterm project for the sixth week of mlzoomcamp](https://github.com/sayalaruano/MidtermProject-MLZoomCamp)

## Day 30: October 26, 2021

**Today's Progress:** I continued with my midterm project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I used XGBoost models as classifiers for my project. I learned how to use GridSearchCV and XGBoost library together for fine-tuning models' hyperparameters. Also, I evaluated all models using different performance metrics, and obtained the best model. However, my final model is overfitting, so I need to try to solve this problem in the coming days.

**Thoughts:** It was interesting to learn how to integrate GridSearchCV and XGBoost, obtaining various scoring metrics for all models with cross validation at the same time. Also, I found that all my models were overfitting, so I need to think how to solve this problem, maybe for the next project.

**Link of Work:**

* [GitHub repository of my midterm project for the sixth week of mlzoomcamp](https://github.com/sayalaruano/MidtermProject-MLZoomCamp)

## Day 31: October 31, 2021

**Today's Progress:** I continued with my midterm project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I made python scripts for training the best machine learning model (random forest) and predicting molecules that bind to betalactamases and are the active ones.

**Thoughts:** In general, I had a good progress. However, I was stuck with the python script to train my model because the input data had two indices, and I could not figure it out this problem for many hours. Also, I had problems with codification of JSON files in my web service with flask.

**Link of Work:**

* [GitHub repository of my midterm project for the sixth week of mlzoomcamp](https://github.com/sayalaruano/MidtermProject-MLZoomCamp)

## Day 32: November 1, 2021

**Today's Progress:** I continued with my midterm project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I created a Python virtual environment with pipenv to manage libraries and dependencies. Also, a Dockerfile allowed to manage SO requirements, python packages,among other specifications to run the web service for predicting molecules that bind to betalactamases and are the active ones.

**Thoughts:** The problem that I had with codification of JSON files was associated to duplicated indices, so I will fix this error leaving one single index. Also, I learn how to install java in a docker image for running PaDEL software.

**Link of Work:**

* [GitHub repository of my midterm project for the sixth week of mlzoomcamp](https://github.com/sayalaruano/MidtermProject-MLZoomCamp)

## Day 33: November 2, 2021

**Today's Progress:** I continued with my midterm project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I deployed my Flask's web service in the cloud using Heroku following [a tutorial](https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-heroku.md) provided by one of the members of the Machine Learning Zoomcamp

**Thoughts:** I enjoyed doing the deployment of my web service in the cloud with Heroku for free. I think that this alternative is very efficient and helpful.

**Link of Work:**

* [GitHub repository of my midterm project for the sixth week of mlzoomcamp](https://github.com/sayalaruano/MidtermProject-MLZoomCamp)

## Day 34: November 6, 2021

**Today's Progress:** I evaluated two projects from classmates of the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). One of the projects was about a classification model that predicts whether a Spotify's track would be a 'Hit' or not, while the other had no information in its repo.

**Thoughts:** I enjoyed evaluating my peers, these projects were interesting and I learned a couple of things such as the sklearn `feature_selection.f_classif` class for calculating ANOVA F-values between features and target variable to select the most relevant features.

**Link of Work:**

* [GitHub repository of midterm project from my peer](https://github.com/ashok-arora/ML-Zoomcamp/tree/main/mid_term_project)

## Day 35: November 7, 2021

**Today's Progress:** I evaluated one project from a classmate of the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). The project was about a regression model that predicts house prices in Iowa.

**Thoughts:** I enjoyed evaluating my peer, and I learned about `Pipelines`, `PCA`, and `ColumnTransformer` sklearn classes, which were a handful way to synthesize code to create th ML model. 

**Link of Work:**

* [GitHub repository of midterm project from my peer](https://github.com/rparthas/data/blob/master/zoomcamp/midterm/)

## Day 36: Nomvember 10, 2021

**Today's Progress:** I watched the [week 10 ML Zoomcamp Office Hours](https://youtu.be/wWBm6MHu5u8). In this session, we reviewed information about the k neartest neighbors algorithm and sklearn Pipelines.

**Thoughts:** I enjoyed learning about both the topics of this lesson. I think that sklearn pipelines are useful for writing readable and summarized code.

**Link of Work:**

* [Youtube video of this lesson](https://www.youtube.com/watch?v=jT0JTlPsAQ0).
