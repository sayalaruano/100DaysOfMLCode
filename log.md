# 100 Days Of ML Code - Log

Hi! I am Sebastián, a Machine Learning enthusiast and this is my log for the 100DaysOfMLCode Challenge.

## Day 1: September 10, 2021

**Today's Progress** : I have completed the [first week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/01-intro). 
In general, I learnt about concept of ML with an example of prediction of car's prizes, differences between ML vs Rule-bases systems, supervised ML tasks, the 
CRISP-DM methodology for organizing ML projects, model selection process, and a quick recap of numpy, linear algebra and pandas. 

**Thoughts** : I liked how the general concepts of ML were presented in the mlzoomcamp, which had examples for each section, and it clarified all the contents. 
I already knew most of the content from recap of numpy, linear algebra and pandas, but it was a nice summary. I enjoyed with the homework, although the 
exercises were not very difficult. However, this review was useful to remember all the concepts that I will use later in the course and other ML projects. 

**Link of Work:** 
* [NotesDay1](Notes/NotesDay1.md)
* [Jupyter notebook of the homework of the first week of mlzoomcamp](Intro_ML/Homework_week1_mlzoomcamp.ipynb)

## Day 2: September 11, 2021

**Today's Progress** : I have completed the first chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). 
In general, I learnt about concept the of ML, why these models are important and some examples, types of ML systems, the main challenges of ML, the importance of 
splitting a ML model in training, validation and test datasets, and other relevant general concepts. 

**Thoughts** : It was interesting the list of main challenges of ML models and some strategies for avoiding them. Also, I realized that the best model obtained after evaluating the validation set, is trained 
with train+val datasets, and what is the importance of cross-validation strategy. Another interesting idea was the *No Free Lunch Theorem* because it reflect that we need to make 
assumptions about the data to choose a few reasonable models, instead of testing all of them. Something that I did not understant was the **train-dev set**. 

**Link of Work:** 
* [NotesDay2](Notes/NotesDay1.md)

## Day 3: September 12, 2021

**Today's Progress** : I have completed the second chapter of the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). 
In general, I learnt about the steps followed to make a complete ML project, including understanding of the problem, data obtention, exploratory data analysis, 
data cleaning and pre-processing, feature engineering, model's training and evaluation, cross-validation, fine-tuning of model's hyperparameters, model deployment. 

**Thoughts** : This chapter was full of a lot of useful concepts and advices. Some interesting ideas an concepts were: l1 and l2 norms, sklearn classes for data 
pre-processing and pipelines, general sklearn design of classes, cross-validation, GridSearchCV and RandomizedSearchCV, and the code examples to perform all the analysis. 

**Link of Work:** 
* [NotesDay3](Notes/NotesDay3.md)

## Day 4: September 13, 2021

**Today's Progress** : I watched the [first part of the lecture Traditional Feature-based Methods](https://www.youtube.com/watch?v=3IS7UhNMQ3U&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=4) 
from the Stanford's course [Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/). This lecture covered the graph's features that can be used for 
training ML models, particularly those derived from nodes. The information-based features included node degree and some centrality measures such as eigenvector, 
betweenness, and closeness centralities. The structure-based features were node degree, clustering cofficient, and graphlet degree vectors. All f these features can 
be used for making predictions of unknown labels from certain nodes. 

**Thoughts** : The lecture had a lot of useful concepts of graph features used for training ML models, particularly the node features. I enjoyed reminding some concepts 
of network science such as centrality measures, clustering coefficient, and graphlets. However, I am intrigued about the way by which all of these features are 
converted to data that the ML model can interpret. 

**Link of Work:** 
* [NotesDay4](Notes/NotesDay4.md)

## Day 5: September 14, 2021

**Today's Progress** :I have completed half of the content of the [second week material of the mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/02-regression). 
In general, I learnt about data preparation, exploratory data analysis, setting up the validation framework, and the application of linear regression model for predicting car prices. Also, we 
understand the internals of linear regression. 

**Thoughts** : I enjoyed the videos of this session, especially the understanding of linear regression model in its vectorized form, and how it can be solved by 
finding the vector of weights or coefficients form the Normal equation. 

**Link of Work:** 
* [NotesDay5](Notes/NotesDay5.md)
