# Notes Week1 - MLOps Zoomcamp

## 1.4 Course overview

Notebooks are usually intended for experiments, which could made the code a mess. So, the notebook's code needs to be improved by being more modular using a python script.

Also, by using notebooks we lost the performance of the previous models and their history. So, we should store the logs of models into a experimental tracker to compare their performances.

Another problem with notebooks is that when we save the file of the best model, we cannot be sure about the performance and other parameters. So, we should save the models into a log file called model registry, which could save the data of all models along with the experiment tracker.

The module 2 about experiment tracking will cover the ways in which we can save useful information of the models using MLflow.

The module 3 about machine learning pipelines will explain how to divide the process into different steps, allowing to optimize the creation and training of models. Using pipelines allows to easily re-execute the code with different parameters.

The output of the pipeline is a model, which needs to be put into a ML service. The different ways of serving an ML model are covered in the module 4.

Also, an important part is the monitoring part, which check if the deployed model is doing well, and correct potential errors. These details are covered in the module 5.

The module 6 about best practices provides ways to automate the processes in ML models. Finally, the module 7 about processes is about how people work together to maintain ML projects.

## 1.5 MLOps maturity model

The hightest maturity of MLOps implies that a model could automatically create and deploy a new model when there are errors, which removes humans from this process. The content of this video was inspired in [this article by Microsoft](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model).

### Level 0 - No MLOps automation

A jupyter notebook with no proper pipelines, experiment tracking, metadata, among other features. So, in this level there is no automation at all.

### Level 1 - DEVOps, but no MLOps

There is some level wih automation. Releases are automated for deploying. People use the best practices of software engineering such as Unit/integration tests, CI/CD, operational metrics, among others. But these standards are not specific for machine learning. It's difficult to reproduce a model and there are not experiment tracking.

### Level 2 - Automated training

There is a training pipeline, and it's possible to parametrize the model. Also, there are experiment tracking, model registry, and low friction deployment. There are multiple models running (2 or more cases).

### Level 3 - Automated deployment

Don't need a human to deploy a model or it's easy to do this process. Here, there is A/B test, which means to search for the best models among various options. Also, model monitoring is applied here.

### Level 4 - Full MLOps automation

Automated training, re-training and deployment together in one place. This is the hightest level of maturity.

When we are exploring a project, we could be in the level 0, but once the project needs to be improved, we could advance to the higher levels. Most of the projects are ok with level 2 and don't need to be on levels 3 or 4.
