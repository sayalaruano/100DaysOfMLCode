# NotesDay1

## Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow chapter 1 - The Machine
Learning Landscape

### What is ML ?
In summary, ML is the science (and art) of programming computers so they can learn from data.

### Why Use Machine Learning?
The ML systems do not need to be programmed with specific rules to perform certain tasks, instead these 
models detect patterns in data and automatically perform their tasks. 

In certain problems, ML models are shorter, easier to maintain, and more accuarte than the rule based systems, especially 
in complex problems that can not be solved with traditional algorithms. Also, ML strategies can help us to understand 
the problems in a better way. 

**Data mining: **Discover patterns that are not immediately apparent in large amounts of data. 

In brief, ML models are great for: 
* Problems for which the current solutions require a long list of rules. 
* Complex problems that can be solved with the traditional approaches. 
* Fluctuating environments 
* Problems with large amounts of data. 

### Types of ML systems 
#### Based on the requeriment of human supervision 
* **Supervised:** training set includes the desired solutions or **labels**. 
	* Classification: category identification. 
	* Regression: predict a target numeric value. 
**Attribute vs feature:** the first one is a data type (e.g. mileage), and the other corresponds to an attribute plus its value. 
**Examples:** k-neartest neighbors, linear regression, logistic regression, SVM, DT, RF, NN

*  **Unsupervised:** training data is unlabeled. 
	* Clustering: Communities detection. Some examples: k-means, DBSCAN, Hierarchical clustering (subdivide each group into smaller groups).
	* Anomaly and novelry detection: AD refers to automatically removing outliers from a dataset, so these algorithms are trained with many normal instances, 
	and they can recognize anomalous data. ND models detect new instances that are different from most of the data. Some examples: One-calss SVM, Isolated forest. 
	* Visualization and dimensionality reduction (DR): The visualization algorithms try to preserve as much structure as they can, while DR models
	simplofy the data without losing too much information. Some examples: PCA, Kernel PCA, Locally linear embedding, t-Distributed Stochastic Neighbor Embedding (t-SNE).
	* Association rule learning: The goal is to dig into large amounts of data and discover interesting relations between attributes. Some examples: Apriori, Eclat.  
* **Semisupervised:** there are plenty of unlabeled instances, and few labeled entries. Some examples include photo-hosting services such as Google photos.
An example could be deep belief networks. 
* **Reinforcement Learning:** the model is an agent that can perceive the environment, select and perform actions, and get rewards in return. Then, the agent learns
the best strategy or policy to obatin the most reward. An example is AlphaGo. 

#### Based on the capacity to learn incrementally 

* **Batch learning**: the system can not learn incrementally, it must be trained using all the available data. These models are tipically trained offline. This
strategy is advantageous when the amount of data is huge. 
 
* **Online learning:** the system is trained incrementally by feeding it data instances sequentially, either individually or in small groups called
**mini-batches**. This is great for systems that receive data as a continuous flow, and for working with datasets that can not fit in machine's memory. The 
**learning rate** is an important parameter, which measures how fast the system should adapt to changing data. A challenge with this strategy is the monitoring 
of incoming data. 

#### Based on the ability to generalize 

* **Instance based systems:** the system learns examples by heart, and a similarity measure is used to generalize the learned patterns to new data  
* **Model based systems:** there is a model trained with examples, and it is applied to make new predictions. These models apply a performance measure to 
evaluate how well the model is working, which can be a utility or fitness function (how good the model is) or a cost function (how bad is the model performance). 

**Inference:** make predictions on new cases. 

### Main challenges of ML
* Insufficient Quantity of Training Data
* Nonrepresentative Training Data - sampling bias
* Poor-Quality Data - error, outliers or noise
* Irrelevant Features - feature engineering consists of two steps: feature selection adn feature extraction (combine featutes to produce a useful one) 
* Overfitting the Training Data - the model performs well on the traning data, but it does not generalize well. Constraning a model to make it simpler 
and reduce risk of overfitting is called **regularization**.
* Underfitting the Training Data - model is too simple to learn the underlying structure of data. Some option for overcome underfitting are: select a more
powerful mode, feature engineering, reduce regularization.

### Testing and Validating
**Generalization error:** error rate on new cases, which measures how well the model will perform on new instances it has never seen before. 

**Holdout validation:** hold out part of the training set to evaluate several candidate models and select the best one. The idea is training multiple models 
with various hyperparameters on the reduced training set, and select the model that performs best on the validation set. After this process, the best model is
trained on the full training set (train+val). 

**Cross-validation:** each model is evaluated once per validation set after it is trained on the rest of the data. Then, averaging out all the evaluations of
a model the performance measure is more accurate. Its drawback is that the training time is multiplied by the numer of validation sets. 

**Data mismatch:** hold out some of the traning set in a new set called **train-dev set**, allowing to identify if the model is performing bad because 
overfitting on training data or due to a mismatch between training and production data. 

**No free lunch theorem:** if there are not assumptions about the data, there is not reason to prefer one model over any other. There is no model that is 
a priori guaranteed to work better, and the only way to know for sure which model is the best is to evaluate them all. This is not possible, so it is required 
to make assumptions about the data and choose a few reasonable models. 
  





