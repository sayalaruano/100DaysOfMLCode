# NotesDay37 - Machine Learning Zoomcamp tenth week

## 8.1 Description og the problem: Fashion classification

This session is the first one in which the input data is not tabular, instead we will work with images. The project of this session is a clothing classifier, which distinguish among ten types of clothes, so this is a multi-class classification task. 

The fifth week of Machine Learning Zoomcamp is about
deployment of ML models. We will deploy the churn model prediction developed the last weeks as a web service. In general, we need to save the model from the Jupyter notebook and load it into a web service, for which we will use [Flask Python library](https://flask.palletsprojects.com/en/2.0.x/). Also, we will use [Pyenv](https://github.com/pyenv/pyenv) to create a Python environment to manage software dependencies, and [Docker](https://www.docker.com/products/docker-desktop) to create a container for handling system dependencies. Ultimately, we will deploy the container in the cloud with AWS EB.

The Jupyter notebook with code of the churning prediction model to be deployed is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-05-deployment/05-deploy.ipynb).

## 8.2 TensorFlow and Keras

We used [Pickle library](https://docs.python.org/3/library/pickle.html) to save and load a machine learning model. In general, pickle allows us to save Python objects. When we load a model, we need to guarantee that all the required libraries are installed in the working Python environment.

For training a machine learning model and making predictions multiple times, it is advisable to convert jupyter notebooks into Python scripts.

**Libraries, classes and methods:**

* `open(x, 'yb')` - open a binary file with the name assigned in the x string, which has permission y, which can be for writing ('w') or reading ('r'). Writing permission is used for creating files and reading is applied for loading files. A binary file contains bits instead of text.
* `pickle.dump(x, y)` - Pickle function to save a python object x into a y file.
* `x.close()` - close the x file. It is important to guarantee that the file contains the object saved with pickle.
* `with open(x, 'yb') as y:` - same as `open(x, 'yb')`, but in this case you guarantee that this file will be closed.
* `pickle.load(x)` - Pickle function to load a python object x.

The python scripts for training our model and making predictions are available [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment/code).

## 8.3 Pre-trained convolutional neural networks

A web service is a method for communication between devices over a network. In general, users make requests with some information, and they receive a result processed by the web service.

We created a simple web service that receives a query with a ping address, and it replies with pong message. Then, we queried the ping/pong web service with `curl` and browser.

**Libraries, classes and methods:**

* `Flask('x')` - create a Flask object with x name.
* `@x.route('x', methods=['y'])` - add a declarator (additional utilities) that specifies the address of an x object, and a method y (i.e. GET, POST, etc) for accessing to it.
* `x.run(debug=True, host=x, port=y)` - run a x Flask object in the debug mode with an x host and a y port. This code should be in a main function.

The python script of ping/pong web service is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment/code).

## 5.4 Serving the churn model with Flask

We created a churn service with our machine learning model, which will be available at the `/predict` address. In this way, other services can send requests with information about customers, and receive responses with churn predictions from the web service.

The method associated with the web service was `POST` because it was required to send some information about customers, which is not easy to do with `GET` method. The requests and responses are sent with JSON files, which are quite similar to Python dictionaries.

The `gunicorn` library helps us to prepare a model to be launched in production. This library is not supported in Windows because it needs some Unix dependencies. So, the alternative for Windows is `waitress` library.

**Libraries, classes and methods:**

* `request.get_json()` - Flask utility to obtain the body of a request as a Python dictionary.
* `jsonify()` - Flask utility to convert a Python dictionary into a tJSON file.
* `requests().post()` - method from `requests` library to perform a POST to a web service. The 200 code means that the process was successful.
* `requests().post().json()` - method from `requests` library to perform a POST to a web service and obtain the json response as a python dictionary.
* `gunicorn --bind x y:z` - library for running a model in production stage. The x is the host, y is the address' name, and z is the name of the object that will be launched.
* `waitress --listen x y:z` - library for running a model in production stage. The x is the host, y is the address' name, and z is the name of the object that will be launched.

The python script of the churning web service is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment/code).
