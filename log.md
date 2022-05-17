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

**Today's Progress:** I have completed all the content for the [second week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/02-regression). This material included data preparation, exploratory data analysis, the validation framework, linear regression model, LR vector and normal forms, the baseline model, root mean squared error, feature engineering, regularization, tuning the model, and using the best model with new data. All these concepts were explained using the problem to predict the price of cars.

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

## Day 36: November 10, 2021

**Today's Progress:** I watched the [week 10 ML Zoomcamp Office Hours](https://youtu.be/wWBm6MHu5u8). In this session, we reviewed information about the k neartest neighbors algorithm and sklearn Pipelines.

**Thoughts:** I enjoyed learning about both the topics of this lesson. I think that sklearn pipelines are useful for writing readable and summarized code.

**Link of Work:**

* [Youtube video of this lesson](https://www.youtube.com/watch?v=jT0JTlPsAQ0).

## Day 37: November 11, 2021

**Today's Progress:** I watched the 1st-3rd videos of [week 10 ML Zoomcamp](https://youtu.be/wWBm6MHu5u8). In this session, we reviewed the description of the project, a fashion classifier, learned about TensorFlow and Keras, and used a pre-trained convolutional neural network to make predictions on an image.

**Thoughts:** It was interesting to learn about Tensorflow and Keras, and how to use a pre-trained neural network for making new predictions.

**Link of Work:**

* [NotesDay37_mlzoomcamp_10thweek_1](Neural_Networks/Notes/NotesDay37_mlzoomcamp_10thweek_1.md)

## Day 38: November 12, 2021

**Today's Progress:** I watched the 4th-7th videos of [week 10 ML Zoomcamp](https://youtu.be/wWBm6MHu5u8). In this session, we learned the theoretical fundamentals of CCNs, the concept of transfer learning and how to implement it, learning rate fine-tuning, and how to save the best models by checkpointing.

**Thoughts:** I understood the theory behind CNNs and transfer learning, which were interesting and helpful concepts. Also, I learned how to implement this concept with keras.

**Link of Work:**

* [NotesDay38_mlzoomcamp_10thweek_2](Neural_Networks/Notes/NotesDay38_mlzoomcamp_10thweek_2.md)

## Day 39: November 13, 2021

**Today's Progress:** I watched the 8th-12th videos of [week 10 ML Zoomcamp](https://youtu.be/wWBm6MHu5u8). In this session, we learned how to add inner layers to a pre-trained CNN, some techniques of regularization such as dropout nd data augmentation.
**Thoughts:** I enjoyed learning about the modification of the architecture of a CNN by adding inner layers, and regularization techniques.

**Link of Work:**

* [NotesDay39_mlzoomcamp_10thweek_3](Neural_Networks/Notes/NotesDay39_mlzoomcamp_10thweek_3.md)

## Day 40: November 19, 2021

**Today's Progress:** I watched the [week 11 ML Zoomcamp Office Hours](https://www.youtube.com/watch?v=1WRgdBTUaAc). In this session, we learned how to implement a CNN from scratch, including some key concepts of CCNs such as kernel size, max pooling, among others. Also, we learned how to implement a multi-layer perceptron neural network to work with tabular data. Finally, we learned about functional and sequential styles to create neural networks with keras.

**Thoughts:** I enjoyed learning about all the topics of this lesson. Particularly, it was interesting to learn how CNNs identify patterns in images using kernels. Also, I find interesting the idea of multi-layer perceptron to work with tabular data.

**Link of Work:**

* [Youtube video of this lesson](https://www.youtube.com/watch?v=1WRgdBTUaAc).

## Day 41: November 20, 2021

**Today's Progress:** I completed the homework for the [tenth/eleventh week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/homework.md). In this homework, we built a CNN model to classify images of dogs and cats. We used this [dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
**Thoughts:** I enjoyed learning how to implement a CNN from scratch (without a pre-trained model), including the addition of inputs, convolutional, pooling, and dense layers. Also, it was great to use image data loaders from keras.

**Link of Work:**

* [Jupyter notebook of the homework for the tenth/eleventh week of mlzoomcamp](Neural_Networks/Notebooks/homework_mlzoomcamp_week10_11_CNN.ipynb)

## Day 42: November 25, 2021

**Today's Progress:** I watched the [week 12 ML Zoomcamp Office Hours](https://youtu.be/wWBm6MHu5u8). In this session, we reviewed the homework of the 10th week. In this homework, we built a CNN model to classify images of dogs and cats. We used this [dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

**Thoughts:** All the answers of this homework were correct.

**Link of Work:**

* [Youtube video of this lesson](https://www.youtube.com/watch?v=plqTzspLU8Y).

## Day 43: November 26, 2021

**Today's Progress:** I watched the 1st-4th videos of [week 12 ML Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/09-serverless). In this session, we reviewed the description of the project, a fashion classifier, learned about TensorFlow Lite, and AWS Lambda to deploy a mode without a server.

**Thoughts:** It was interesting to learn about Tensorflow Lite as an alternative to export a lighten version of a DL model. Also, I enjoyed learning about AWS Lambda for deploying a DL model without a server.

**Link of Work:**

* [NotesDay43_mlzoomcamp_12thweek_1.md](Serverless_Deep_Learning/Notes/NotesDay43_mlzoomcamp_12thweek_1.md)

## Day 44: November 27, 2021

**Today's Progress:** I watched the 5th-8th videos of [week 12 ML Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/09-serverless). In this session, we created the docker image with all the requirements, the lambda function in AWS Lambda, and we used the AWS API Gateway to expose the lambda function as a web service.

**Thoughts:** I enjoyed learning how to create a docker container with all the requirements to create a lambda function and then publish it as a web service using AWS Lambda and the AWS API Gateway.

**Link of Work:**

* [NotesDay44_mlzoomcamp_12thweek_1.md](Serverless_Deep_Learning/Notes/NotesDay44_mlzoomcamp_12thweek_2.md)

## Day 45: November 28, 2021

**Today's Progress:** I completed the homework for the [12th week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/09-serverless/homework.md). In this homework, we deployed a serverless CNN model to classify images of dogs and cats. For this purpose, we created the model with TensorFlow Lite, prepared a docker container with the model, a python script with the lambda function and the rest of the requirements, and we published this container in AWS Lambda.

**Thoughts:** It was interesting to work on a serverless version of the CNN model that we created in the last homework. I liked how to create the docker container with all the requirements for the model, which is an efficient way to address the potential dependencies issues. Also, it was great to know how to work with AWS Lambda.

**Link of Work:**

* [Jupyter notebook of the homework for the 12th week of mlzoomcamp](Serverless_Deep_Learning/Homework_ml-zoomcamp_twelfth_week/Homework_ml-zoomcamp_twelfth_week.ipynb)

## Day 46: December 3, 2021

**Today's Progress:** I started my capstone project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project).  I decided to work in machine/deep learning binary classifiers to predict the activity of antimicrobial peptides using this [dataset](biocom-ampdiscover.cicese.mx/dataset). I did the README of my project, an explored similar resources. I took a [notebook from Dataprofessor](https://github.com/dataprofessor/peptide-ml/blob/main/Antimicrobial_Peptide_QSAR.ipynb) about this topic as the starting point of my work.

**Thoughts:** I enjoyed doing my project because drug discovery is the field that capture my interest for postgraduate studies, and it was really nice to learn more about it. I liked to learn about basic concepts of drug discovery, and how to apply this knowledge in a machine learning project.

**Link of Work:**

* [GitHub repository of my capstone project](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp)

## Day 47: December 9, 2021

**Today's Progress:** I continued with my capstone project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I obtained 10 feature matrices using different molecular descriptors using [Pfeature](https://github.com/raghavagps/Pfeature). Also, I did the Exploratory data analysis of all the matrices, run many ML classifiers with LazyPredict, and chose the best ML models and feature matrices.

**Thoughts:** I liked to use Pfeature to extract molecular descriptors from AMPs, cd-hit to delete redundant AMPs, and LazyPredict to calculate many classifiers in an easy and fast way.

**Link of Work:**

* [GitHub repository of my capstone project](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp)

## Day 48: December 10, 2021

**Today's Progress:** I continued with my capstone project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I identified the best feature matrix and the best ML classifier, which were the ExtraTreesClassifier with `max_depth` of 50 and `n_estimators` of 200 as parameters, and `Amino acid Composition` (aac_wp) as feature matrix. I evaluated the performance of the best model on the test and external dataset with metrics such as accuracy, precision, recall, f1 score, and MCC. Surprisingly, the evaluation results of our model were close to the best ones reported in this [article](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c00251).

**Thoughts:** I was surprised about the great results of our best model, which performed as well as the state of the art methods to predict AMPs. Also, it was interesting that the Amino Acid Composition was the best feature matrix because of its simplicity.

**Link of Work:**

* [GitHub repository of my capstone project](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp)

## Day 49: December 12, 2021

**Today's Progress:** I continued with my capstone project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I finished the jupyter notebook with EDA, feature matrices calculation, ML binary classifiers, hyperparameter fine-tuning, and selection of the best models. Also, I exported the best model as a web service using Flask, created a Pipenv environment to manage python dependencies, and a Docker file to handle OS requirements and dependencies.

**Thoughts:** It was difficult to handle the dependencies to install Pfeature, which I used to calculate the molecular features of AMPs' sequences. There was a problem with Debian OS to install this python library, so I needed to search for a docker image with python and Ubuntu.

**Link of Work:**

* [GitHub repository of my capstone project](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp)

## Day 50: December 13, 2021

**Today's Progress:** I finished my capstone project for the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/07-midterm-project). I deployed the best ML binary classifier to predict AMPs' activity as a web service in Heroku using the Docker image I previously created. Also, I completed and reviewed all the sections of README file for the GitHub repository.

**Thoughts:** I liked that Heroku allows to deploy web services with a Docker image, which makes this work easy. However, in this project I struggled a bit with the creation of the Docker image because some errors to install Pfeature python library with Docker images based on Debian OS.

**Link of Work:**

* [GitHub repository of my capstone project](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp)

## Day 51: December 17, 2021

**Today's Progress:** I evaluated one classmate's project of the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp). This project used the well-known titanic dataset, in which the aim is to predict the survival of passengers considering some of these features. This project has all the stages for a data science project and tried multiple ML models, including logistic regression, decision trees, random forest, xgboost, and neural networks.

**Thoughts:** First, I liked the GitLab interface to store code projects. Regarding to the project, it was interesting that the best model was a decision tree instead of random forest or XGBoost, which usually perform better with tabular data. Also, the performance metrics were very high compared to other projects I have worked on.

**Link of Work:**

* [GitHub repository of my classmate's project](https://gitlab.com/hda-at/ml_zoomcamp_capstone)

## Day 52: December 18, 2021

**Today's Progress:** I evaluated one classmate's project of the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp). This project was about the creation of ML models to predict if a customer will purchase a product or not.  This project has all the stages for a data science project and tried multiple ML models, including xgboost, neural networks, and some tree-based classifiers.

**Thoughts:** In this project, there were some sklearn classes I haven't worked with, including feature_selection to select the best features, MLPClassifier to work with neural networks, powertransformer to create pre-processing pipelines, and new ML classifiers such as BaggingClassifier, LGBMClassifier, and VotingClassifier. Also, it was interesting the use of optuna library for hyperparameter tuning.

**Link of Work:**

* [GitHub repository of my classmate's project](https://github.com/snikhil17/Customer_Shopping_Intention)

## Day 53: December 19, 2021

**Today's Progress:** I evaluated one classmate's project of the [mlzoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp). This project was about the creation of ML models to predict CO2 emissions of vehicles considering certain technical characteristics. This project has all the stages for a data science project and tried multiple ML models, including linear regression, ridge regression, random forest, xgboost, and neural networks.

**Thoughts:** I liked the task of the project, which took some features of vehicles, and predict their CO2 emissions. This can be helpful when you want to buy a vehicle. Also, I learned how to use the *TqdmCallback* class for tracking the progress of learning from NN models trained with Keras.

**Link of Work:**

* [GitHub repository of my classmate's project](https://github.com/DZorikhin/co2_emissions)

## Day 54: December 20, 2021

**Today's Progress:** I started the second course of the Coursera DeepLearning.AI specialization - Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization. I watched the first five videos of week 1.

**Thoughts:** It was interesting to learn the new conventions to split up the datasets for DL models. Also, I liked the tips provided to identify high variance or high bias models, and how to prevent these issues. Finally, I enjoyed to learn the intuition of regularization techniques.

**Link of Work:**

* [DeepLearningAI_2ndcourse_1thweek.md](Neural_Networks/Notes/DeepLearningAI_2ndcourse_1thweek.md)

## Day 55: December 27, 2021

**Today's Progress:** I finished the first week of the first course of the Coursera DeepLearning.AI specialization - Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization. I watched all the videos and completed the programming assignments.

**Thoughts:** I liked to learn about different regularization methods for DNN, including L2 regularization, dropout, data augmentation, and early stopping. Also, I found interesting the techniques applied to optimize the DL models such as normalization of inputs, how to handle possible problems with very high or very low values of gradients, and how to identify if the back propagation method is working well.

**Link of Work:**

* [DeepLearningAI_2ndcourse_1thweek.md](Neural_Networks/Notes/DeepLearningAI_2ndcourse_1thweek.md)

## Day 56: December 29, 2021

**Today's Progress:** I watched all the videos and completed the quiz of the second week of the first course of the Coursera DeepLearning.AI specialization - Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization. This week was about optimizaton algorithms to improve the DL algorithms.

**Thoughts:** It was intteresting to learn about various implementations of gradient descend with improvements that makes this algorithm to run faster. Also, I nejoyed to learn about the strategy of learning rate decay and the local minimmums on high dimensional problems.

**Link of Work:**

* [DeepLearningAI_2ndcourse_2ndweek.md](Neural_Networks/Notes/DeepLearningAI_2ndcourse_2ndweek.md)

## Day 57: January 06, 2021

**Today's Progress:** I watched the 1st-3th videos of [week 13 ML Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/10-kubernetes). In these videos, we learned about the general idea of using kebernetes and tensorflow serving to deploy models in production environments. Also, we converted the tensorflowlite model to the tenforflow serving format, and we created the gateway or pre-processing service that allows to communicate the web service with tensorflow serving.

**Thoughts:** It was interesting to use tensorflow serving and the flask gateway to communicate the DL model with the web service.

## Day 58: January 07, 2021

**Today's Progress:** I watched the 4th-6th videos of [week 13 ML Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/10-kubernetes). In these videos, we created the docker images for the TensorFlow serving and the gateway, and we used docker decompose to run both docker containers in the same network to communicate with each other. Also, we reviewd the general aspects of Kubernetes, which helps to deploy docker images to the cloud, manage them, scale up requests, and create multiple instances of the web services. Finally, we deployed an application into a Kubernetes cluster.

**Thoughts:** It was interesting to learn about Docker decompose to run multiple docker containers simultaneously in the same network to communicate with each otherand. Also, I liked Kubernetes to deploy containers into the cloud in a well structured manner.

## Day 59: January 08, 2021

**Today's Progress:** I watched the 7th-9th videos of [week 13 ML Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/10-kubernetes). In these videos, we set up a kubernetes cluster the docker images for the TensorFlow serving and the gateway using kubectl and Kind. Also, we deployed the kubernetes cluster into EKS, an AWS' Kubernetes service.

**Thoughts:** It was interesting to learn how to use Kind and kubectl to set up a kubernetes cluster with different services and pods. Also, I liked to learn about EKS, an AWS' Kubernetes service for deploying models organized on a kubernetes cluster into the cloud.

## Day 60: January 10, 2021

**Today's Progress:** I completed the homework for the [fourth week of mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/10-kubernetes/homework.md). In this homework, we created a local Kubernetes cluster for a churning classifier using kubectl and Kind.

**Thoughts:** I enjoyed applying the knowledge of this session about Kubernetes in a previous machine learning classifier of the course.

**Link of Work:**

* [Markdown file of the homework for kubernetes](Deployment/Homework_mlzoomcamp_kubernetes.md)

## Day 61: January 13, 2021

**Today's Progress:** I watched all the videos and completed the quiz from third week of Coursera DeepLearning.AI specialization second course - Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization. This week was about hyperparameters, how to effectivetely do hyperparameter tuning by using an appropriate scale, batch normalization, multi-class calssification tasks with softmax, and a brief introduction to tensorflow.

**Thoughts:** It was interesting to learn that random sampling with a proper scale is an effective way to choose hyperparameters. Also, I learnt about batch normalizaton, and how to implement this algorithm. Finally, we reviewed a brief introduction to tensorflow.

## Day 62: January 16, 2021

**Today's Progress:** I completed the programming assignment of third week of Coursera DeepLearning.AI specialization second course - Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization. With this assignment, I also finished this course. The programming assignment was about a simple DL model to predict image lables of sign language digits using TensorFlow.

**Thoughts:** It was interesting how to program a simple neural network with TensorFlow, which is quite easier than doing this process from scratch using numpy.

**Link of Work:**

* [Notebook of my programming assignment of the thrid week's Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization course](Neural_Networks/Deep_learning_Specialization_Coursera/Improving_Deep_Neural_Nets_Hyperparameter_Tuning_Regularization_Optimization/Week3/Tensorflow_introduction.ipynb)

## Day 63: January 17, 2021

**Today's Progress:** I attended the [week 1 Data engineering Zoomcamp Office Hours](https://youtu.be/wWBm6MHu5u8). In this session, we reviewed the overview of the course, and we talked about general questions.

**Thoughts:** I am really excited with this course because most of the contents are new for me.

**Link of Work:**

* [Youtube video of this session](https://youtu.be/bkJZDmreIpA)

## Day 64: January 21, 2021

**Today's Progress:** I watched the 1th-7th videos of [week 1 DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp/tree/main/week_1_basics_n_setup). In these videos, we learned about the basics of docker, SQL, sqlalchemy, Postgres, and pgAdmin. We ran a docker container with Postgres, ingested the data into the database with the sqlalchemy python library , ran a container with pgAdmin, and made queries to the Postgres database with pgAdmin.

**Thoughts:** I liked to remember the basics of SQL using Postgres as the database management system and pgAdmin to administrate SQL queries to the relational database. Also, it was great to use docker compose to run Postgres and pgAdmin containers in a single terminal and withot many parameters at the buiding of docker images. I also enjoyed learning how to ingest data into a relational database using sqlalchemy python ibrary.

**Link of Work:**

* [Videos of this session](https://www.youtube.com/playlist?list=PL3MmuxUbc_hJed7dXYoJw8DoCuVHhGEQb)

## Day 65: January 22, 2021

**Today's Progress:** I watched the 8th-10th videos of [week 1 DE Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp/tree/main/week_1_basics_n_setup). In these videos, we learned about terraform to manage the infrastructure and resources on Google Cloud Platform.

**Thoughts:** I liked to learn how to manage resources in cloud providers using Terraform. These topics are entirely new for me, so I am trying to learn as much as I can.

**Link of Work:**

* [Videos of this session](https://www.youtube.com/playlist?list=PL3MmuxUbc_hJed7dXYoJw8DoCuVHhGEQb)

## Day 66: January 23, 2021

**Today's Progress:** I completed the homework for the [first week of dezoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp/blob/main/week_1_basics_n_setup/homework.md). In this homework, we created terraform infrastructure, and we practiced some queries with SQL.

**Thoughts:** I enjoyed remembering SQL, and it was great to put in practice the oncepts of terraform and Google Cloud Platform.

**Link of Work:**

* [Markdown file of the homework](Data_Engineering/Docker_SQL_GCP_Terraform/homework_de_zoomcamp_week1.md)

## Day 67: April 4, 2022

**Today's Progress:** I have joined the 30DaysOfStreamlit challenge, and completed the first 3 days assignments. I set up my development environment for streamlit with conda, ran the streamlit demo app, chose VS Code as my IDE, created my hello world streamlit app, and learned about the `st.button` component to create a button widget.

**Thoughts:** I enjoyed learning about the basic ideas of sreamlit. I understood all the contents, except for the on_click parameter of the `st.button`.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/)

## Day 68: April 5, 2022

**Today's Progress:** I commpleted the days 4 and 5 of the 30DaysOfStreamlit challenge. I watched a video by Ken Jee for creating a Dashboard with streamlit, and learned about the `st.write` function for writing text and arguments to streamlit apps.

**Thoughts:** I enjoyed learning about creating a dashboard with streamlit, and I was amazed about how easy it was. I will try to create my own dashboard about the Sars-Cov2 variants of interests in Europe. Also, it was interesting how to use the write function to display different data such as md text, numbers, dataframes, and plots.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/)

## Day 69: April 6, 2022

**Today's Progress:** I commpleted the day 6 of the 30DaysOfStreamlit challenge. I learned about the basics of git and github, and how to create a github repository, which allows to deploy streamlit apps in the cloud.

**Thoughts:** It was interesting to remind the basics of git and github, and how they can be used to deploy streamlit apps to the cloud.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/)

## Day 70: April 7, 2022

**Today's Progress:** I commpleted the day 7 of the 30DaysOfStreamlit challenge. I created and deployed a dashboard of weekly reports of SARS-CoV2 variants in Germany from 2020 until now with streamlit.

**Thoughts:** I enjoyed deploying my first streamlit app to the cloud. It was easy and I will try to improve this project in the coming days.

**Link of Work:**

* [Strealit app](https://share.streamlit.io/sayalaruano/dashboard_sars-cov2_variants_europe/main/st_dashboard_1country.py)

## Day 71: April 8, 2022

**Today's Progress:** I commpleted the days 8 and 9 of the 30DaysOfStreamlit challenge. I learned about the st.silder and st.line_chart functions.

**Thoughts:** It was interesting how to create sliders with different date types like integers, float, and datatime. I realized that for creating sliders with any data type I should use st.selct_slider. Also, I enjoyed learning about the creation of line charts in streamlit.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 72: April 11, 2022

**Today's Progress:** I commpleted the days 10 and 11 of the 30DaysOfStreamlit challenge. I learned about the st.slectbox and st.multiselect functions.

**Thoughts:** It was interesting to learn about the select and multiselect streamlit functions. The interface of the multiselect is appealing and very useful for many applications.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 73: April 12, 2022

**Today's Progress:** I commpleted my project about a dashboard of weekly reports of SARS-CoV2 variants in European countries since 2020 until April 2022.

**Thoughts:** I really enjoyed this project. I was amazed at how easy was to create beautiful web apps with streamlit.

**Link of Work:**

* [GitHub repo of the dashboard](https://github.com/sayalaruano/Dashboard_SARS-CoV2_variants_Europe)
* [Web app of the dashboard](https://share.streamlit.io/sayalaruano/dashboard_sars-cov2_variants_europe/main/st_dashboard_allcountries.py)

## Day 74: April 15, 2022

**Today's Progress:** I commpleted the days 12, 13, and 14 of the 30DaysOfStreamlit challenge. I learned about the st.checkbox function, [GitPod](https://www.gitpod.io/) to create a cloud development environment, and [streamlit components](https://docs.streamlit.io/library/components).

**Thoughts:** I enjoyed learning about how to create isolated developing environments using GitPod, which is a very useful concept for many programming projects. Also, it was interesting to learn about streamlit components and how to create one of these components.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 75: April 17, 2022

**Today's Progress:** I commpleted the days 15, 16, and 17 of the 30DaysOfStreamlit challenge. I learned about the st.latex function, how to customize the theme of streamlit apps with a config.toml file, and st.secrets to store confidentail information.

**Thoughts:** I learned to add latex code into streamlit apps, customize the theme of streamlit apps, and store confident information. Also, it was interesting the st.code function to create code cells.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 76: April 18, 2022

**Today's Progress:** I commpleted the day 18 of the 30DaysOfStreamlit challenge. I learned about the st.file_uploader function, which allows to create a widget for uploading files.

**Thoughts:** It was interesting how to use the st.file_uploader function and the pandas input/output functions for displaying a cv file.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 77: April 20, 2022

**Today's Progress:** I commpleted the days 19 and 20 of the 30DaysOfStreamlit challenge. I learned about some functions to customize the layout of streamlit apps, including st.set_page_config(layout="wide"), st.sidebar, st.expander, and st.columns. Also, I attended part of the twitter space about streamlit hosted by [Francesco Ciulla](https://twitter.com/FrancescoCiull4).

**Thoughts:** I enjoyed learning how to customize the layout of an streamlit app with expanders, columns, and sidebars. Also, I learned a lot of useful concepts and ideas from the twiter space.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 78: April 22, 2022

**Today's Progress:** I commpleted the days 21 and 22 of the 30DaysOfStreamlit challenge. I learned about the st.progress and st.form functions. The first one is applied to visualize a progress bar, while the second one allows to group various widget inputs and send all the data with a single click.

**Thoughts:** It was interesting to learn about the st.form because it creates batch submits of various widget inputs. In this way, a streamlit app doesn't rerun everytime the user change a widged, and the app reruns only when the user click on the submit button.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 79: April 26, 2022

**Today's Progress:** I commpleted the days 23, 24, and 25 of the 30DaysOfStreamlit challenge. I learned about the st.experimental_get_query_params and st.session_state functions, and st.cache decorator The first one retrives the query parameters from urls, the second one allows to use session states and callbacks, and the last one improves the performance of streamlit apps.

**Thoughts:** It was interesting to learn that urls have qurey parameters that the st.experimental_get_query_params function could recover. Also, I enjoyed learning about the st.cache decorator that improves the performance of streamlit apps by storing results of functions in the cache. Streamlit controls this process by hashing, referencing each hash to a specific result. Finally, I understood how to use st.sessio_state to share variables between various sessions and manipulate their state using callbacks.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 80: April 27, 2022

**Today's Progress:** I commpleted the days 26 and 27 of the 30DaysOfStreamlit challenge. I learned about how to use APIs in streamlit apps, and creating draggable and resizable dashboards with streamlit elements component.

**Thoughts:** It was interesting to learn about how to use APIs with streamlit using the simple example of the bored API. Also, I loved the streamlit elemments componenets for creating customizable and appealing dashboards.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 81: April 28, 2022

**Today's Progress:** I commpleted the days 28 of the 30DaysOfStreamlit challenge. I learned about the streamlit-shap component for creating [shap](https://github.com/slundberg/shap) plots in streamlit apps.

**Thoughts:** It was interesting how to create appealing plots from machine learning projects in stremlit apps usin the streamlit-shap component. I will use these explainable plots in my future ML projects.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 82: May 1, 2022

**Today's Progress:** I commpleted the day 29 of the 30DaysOfStreamlit challenge. I learned about how to create a zero-shot learning text classifier using Streamlit and HuggingFace's Distilbart-mnli-12-3 model.

**Thoughts:** It was interesting to learn about zero-shot learning, a ML strategy to recognize objects that were not observed during the training process. Also, I enjoyed learning about streamlit_option_menu, streamlit_tags, and st_aggrid streamlit components.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 83: May 2, 2022

**Today's Progress:** I commpleted the day 30 of the 30DaysOfStreamlit challenge. I learned about how to create an streamlit app to extract thumbnail images from YouTube videos.

**Thoughts:** It was interesting to learn how to solve real world problems like the retrieval of thumbail images from youtube videos using streamlit apps.

**Link of Work:**

* [Python scripts with streamlit code](Streamlit/Scripts)

## Day 84: May 16, 2022

**Today's Progress:** I started the MLOPs Zoomcamp course, hosted by the DataTalks.Club community. I attended the first live streaming of the course, and watched the first two introductory videos about the concept of MLOPs and the environment preparation. Also, I created a streamlit dashboard for a project of my owrk about phenology information fo tropical american plant species.

**Thoughts:** It was interesting to learn about MLOPs, and the imortance of using good practices and proper tools to put ML models to production. The application of MLOPs practices allow us to automate work and create robust ML applications.

**Link of Work:**

* [MLOPs Zoomcamp repo](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/)
* [Streamlit dashboard](https://share.streamlit.io/sayalaruano/summary_gbif_results_stapp/main/summary_results_GBIF_stapp.py)
