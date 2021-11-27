# NotesDay44 - Machine Learning Zoomcamp eleventh week

## 9.5 Preparing a Docker image

We created a Docker image with the python script for deploying the DL model in AWS Lambda. The base image is available [here](https://gallery.ecr.aws/lambda/python), and its python version was 3.8. Also, we installed some dependencies and copied the TensorFlow Lite model and the python script with the lambda function. In this Docker image, instead of specifying an entry point, we provide a command with the name of the lambda_handler function.

There was an error regarding the version of TensorFlow Lite available in Linux of the docker file and the AWS Lambda, but it was solved by using a pre-compiled version of this library.

The Docker file, TF Lite model, and lambda function python script are available [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/09-serverless/code).

## 9.6 Creating the lambda function

We publish the Docker image obtained in the last lesson into AWS Lambda, created the lambda function, and tested it.

**Commands and methods:**

* `aws ecr create-repository --repository-name x`- aws ecr command to create a repository with an x name.
* `aws ecr get-login --no-include-email`- aws ecr command to log in into a registry a repository with an x name.
* `docker images`- docker command to list the available images.
* `docker tag x:y z`- docker command to associate an x:y docker image with a z tag.
* `docker push x`- docker command to push an image to an x location.

The Docker file, TF Lite model, and lambda function python script are available [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/09-serverless/code).

## 9.7 API Gateway: exposing the lambda function

We took the Lambda function created in the last session, and exposed it as a web service. For this purpose, we used API Gateway utility from AWS. There, we selected a REST API, and added a method with a POST request.

The Docker file, TF Lite model, and lambda function python script are available [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/09-serverless/code).

## 9.8 Summary

* AWS Lambda is way of deploying models without having to worry about servers.
* Tensorflow Lite is a lightweight alternative to Tensorflow that only focuses on inference.
* To deploy your code, package it in a Docker container.
* Expose the lambda function via API Gateway.

The Docker file, TF Lite model, and lambda function python script are available [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/09-serverless/code).
