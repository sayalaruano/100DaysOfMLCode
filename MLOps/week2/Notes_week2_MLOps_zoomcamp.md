# Notes Week2 - MLOps Zoomcamp

## 2.1 Experiment tracking intro

This lesson us not about A/B testing, instead it's about planning ML experiments. An ML experiment is the entire process and each trial is an experiment run. A run artifact is any file associated with an ML run, and there ir experiment metadata.

Experiment tracking: process of keeping track all the relevant information for an ML experiment. What is relevant depends on the experiment, and it could be the source code, environment, data, model, hyperparameters, and metrics. In general, we include standard metrics and customize them later.

Reasons why experiment tracking is important:

* Reproducibility
* Organization
* Optimization

The more basic experiment tracking scheme could be a spreadsheet. The problems with this strategy are:

* Error prone: data is added manually.
* No standard format: different formats by projects and people.
* Visibility and collaboration

MLflow is an open source platform for the ML lifecycle, which includes the process of maintaining ML models. There are four main modules:

* Tracking
* Models
* Model registry
* Projects

This package allows to organize the experiments, and to keep track of different information from the models such as parameters, metrics, metadata, artifacts, and models. Also, there is extra information such as source code, code version (git commit), start and time, and author.

To use the models part of mlflow, it's required to run the model with a backend.

**Commands:**

* `mlflow ui` - run an mlflow server instance

## 2.2 Getting started with MLflow

In this lesson, we installed the mlflow client and implemented it with a backend into the nyc rides project.

**Commands:**

* `pip install -r requirements.txt` - install python packages using a file with pip.
* `mlflow ui --backend-store-uri sqlite:///mlflow.db` - run an mlflow server instance with a backend.
* `with mlflow.start_run()` - start a run with mlflow.
* `mlflow.set_tag("x", "y")` - add an x category tag with y name associated with a run.
* `mlflow.log_param("x", "y")` - save parameters and hyperparameters about the x name associated with the y file.
* `mlflow.log_metric("x", y)` - keep track of the x name metric associated with the y metric.

## 2.3 Experiment tracking with MLflow

In this lesson, we added parameter tuning to the notebook of the previous lesson, show how to implement this process with MLflow, select the best model, and learn about autolog.

**Hyperopt:** a library to find the best set of hyperparameters.

* **fmin:** minimize the object function, and outputs a minimum value.
* **tpe:** control the logic and run the optimization.
* **hp:** define the search space, the rages for each hyperparameters we use.
* **STATUS_OK:** signal send at the end of each run to tell if the object run well.
* **Trials:** keep track information of each run.

We could use the MLflow dashboard to check the best hyperparameters using the parallel coordinates plot, scatter plot, and contour plot.

There is no rule to choose the best model, it depends on the metrics suitable for each project, but it could be selected by sorting the models with the performance metric.

## 2.4 Model management

Machine learning lifecycle refers to the steps required to build and maintain a ML model. Model management covers experiment tracking, model versioning and deployment, and scaling hardware. After obtaining the best ML model, we should save, monitor its versions, and deploy it somewhere.

We could use a folder system as a basic way to manage our ML models, but this strategy is error-prone, without versioning, and there is no model lineage (parameters, data, and other information).

In this lesson, we learned how to log ML models as artifacts with MLflow using two methods: log_artifact (consider the model as another artifact - not very useful) and log_model (more information is stored).

**Commands:**

* `mlflow.log_artifact(local_path="x", artifact_path="y")` - save a model with an x artifact location and a y path to save the model.
* `mlflow.xgboost.log_model(x, artifact_path="y")` - save an x model and its metadata into a y path.
* `mlflow.pyfunc.load_model("x")` - load an ML model as a python function using the x uri model run path of mlflow.
* `mlflow.xgboost.load_model("x")` - load an ML model as an xgboost object using the x run path of mlflow.

## 2.4 Model registry

After the experiment tracking process, we could register the models that are ready for production. Model registry is not deploying any model, it only ist the models ready for deployment. Some aspects to consider for deploying are the performance metrics, running time, and the sie of the model's file.

MLflow allows to register a model with the "Register Model" button of the dashboard. Then, you could access these version fo the ML model with the "Models" tab. After this process, we could assign the model versions to a category such as staging, production, and archived. Then, we could interact by code with the information of registry models using the MLflow client.

In summary, the model registry is a centralized model store, set of APIs, and a UI, to collaboratively manage the full lifecycle of an MLflow model. It provides a model linage (information about how the model was built), model versioning, stage transitions, and annotations.
