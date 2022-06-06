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

## 3.4 Remote Prefect Orion deployment

In this lesson, we deployed a prefect orion server instance on a AWS hosted virtual machine. We created some security permissions to access the cloud instance, specifically the http with the 4200 port. Then, we run some commands to work with prefect on this environment.

## 3.5 Deployment of Prefect flow

In this lesson, we deployed the prefect flow on a remote prefect orion server. First, we should define a place to storage the flow information. In this case, we used a SubprocessFLowRunner, which means that we only have a Python script (no docker or kubernetes). Then, we created a work queue with an agent that looks for work to do.

It's possible to integrate MLflow with Prefect to create a flow that runs for specific periods of time compare the new models with the previous ones, and decide to update the production model.

**Commands:**

* `prefect storage ls` - verify the storage of a prefect flow.
* `prefect storage create` - define a place to storage information of a prefect flow.
