# Notes Week3 - MLOps Zoomcamp

## 3.1 Negative engineering and workflow orchestration

Workflow orchestration is a set of tools that schedule and monitor the work for a project. In ML projects, we run models, analyze potential errors, and fis these issues. There could be failures in many pasrt of the ML workflow.

Negative engineering means that the coding is designed to behave against failures or unexpected changes Workflow orchestration aims to create features that reduce negative engineering, and provides observability to failures.

## 3.2 Introduction to Prefect 2.0

Prefect is a Python open-source workflow orchestration framework. Prefect Core is the first version of the framework, while Prefect Orion is the second one. The second version of Prefect provides DAG free workflows and transparent and observable orchestration rules.

In this lesson, we converted the jupyter notebook from the previous section to a python script with functions for the main tasks.

## 3.3 First Prefect flow and basics

In this lesson, we converted the python script into a prefect flow. Prefect allows to work with concurrent and sequential flows, and MLflow functions with the sequential processes. In prefect, tasks are the units for monitoring and we should pass functions to these decorators.

**Commands:**

* `@flow` - decorator to assign a flow with prefect.
* `@task` - decorator to assign a task with prefect.
* `mlflow.pyfunc.load_model("x")` - load an ML model as a python function using the x uri model run path of mlflow.
* `mlflow.xgboost.load_model("x")` - load an ML model as an xgboost object using the x run path of mlflow.

## 3.4 Model management

Machine learning lifecycle refers to the steps required to build and maintain a ML model. Model management covers experiment tracking, model versioning and deployment, and scaling hardware. After obtaining the best ML model, we should save, monitor its versions, and deploy it somewhere.

We could use a folder system as a basic way to manage our ML models, but this strategy is error-prone, without versioning, and there is no model lineage (parameters, data, and other information).

In this lesson, we learned how to log ML models as artifacts with MLflow using two methods: log_artifact (consider the model as another artifact - not very useful) and log_model (more information is stored).

**Commands:**

* `mlflow.log_artifact(local_path="x", artifact_path="y")` - save a model with an x artifact location and a y path to save the model.
* `mlflow.xgboost.log_model(x, artifact_path="y")` - save an x model and its metadata into a y path.
* `mlflow.pyfunc.load_model("x")` - load an ML model as a python function using the x uri model run path of mlflow.
* `mlflow.xgboost.load_model("x")` - load an ML model as an xgboost object using the x run path of mlflow.

## 3.5 Model registry

After the experiment tracking process, we could register the models that are ready for production. Model registry is not deploying any model, it only ist the models ready for deployment. Some aspects to consider for deploying are the performance metrics, running time, and the sie of the model's file.

MLflow allows to register a model with the "Register Model" button of the dashboard. Then, you could access these version fo the ML model with the "Models" tab. After this process, we could assign the model versions to a category such as staging, production, and archived. Then, we could interact by code with the information of registry models using the MLflow client.

In summary, the model registry is a centralized model store, set of APIs, and a UI, to collaboratively manage the full lifecycle of an MLflow model. It provides a model linage (information about how the model was built), model versioning, stage transitions, and annotations.
