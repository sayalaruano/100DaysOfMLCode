# NotesDay19 - Machine Learning Zoomcamp fifth week

## 5.5 PyThon virtual environment: Pipenv

When we install a Python package with pip, the system looks up in the `$PATH` variable the location of the pip script in the file system. Then, pip goes to the [python package index](https://pypi.org/), and it installs the latest version of the package.

If we have two services in our system that use different versions of a Python package, we need to separate dependencies for each service to avoid possible conflicts. One way to manage dependencies for different services separately is by **virtual environments**, which has their own Python with specific packages, libraries' versions and dependencies.

There are various alternatives for managing virtual environments, including **virtual env** (venv), **conda**, **pipenv**, **poetry**, ***pyenv**, among others. In this lesson, we learned about pienv. Each pienv virtual environment has a `Pipfile` with the names and versions of packages installed in the virtual environment. `Dev-packages` are meant to be used only in the local system, not in the deployment. Pipenv environments also have a `Pipfile.lock`, a json file that contains versions of packages, and dependencies required for each package.

**Commands:**

* `pipenv install x` - install an x package with a pipenv virtual environment.
* `pipenv install` - install packages stored in the `Pipfile` and `Pipfile.lock`, with their specific versions.
* `pipenv shell` - get in a particular virtual environment.
* `pipenv exit` - exit from a pipenv virtual environment.
* `pipenv run x` - run a command in a pipenv virtual environment without entering before using pipenv shell.  

The scripts and notes of this session are available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-05-deployment/05-deploy.ipynb).

## 5.6 Environment management: Docker

Docker allows to create containers, which are isolated environments with specific system requirements such as OS, libraries, programs, dependencies, among others. A host machine can have many different isolated containers with their particular specifications. The main advantage of Docker is that we can take a container with a service and easily deploy to the cloud.

The specifications of the docker container are stated in the `Dockerfile`, including the base image, instructions for installing libraries, files we need to copy from the hots machine, and other instructions. To communicate the container with host machine, we need to expose the port from the container, and establish the communication between container and hots machine with port mapping. In this way, container makes visible the port of the container to the host machine.

**Commands:**

* `docker run -it --rm --entrypoint -p x` - docker command for running an x image. -it allows to access the terminal, --rm will remove the image from the system after use it, --entry point specifies the entry point (i.e. bash, python, etc), and -p to specify the port of container and host machine.
* `docker build -t x y` - Create a container called x with the specifications of the dockerfile y.
* `pipenv install --system --deploy` - install libraries of the Pipfile to the Python system, without creating a virtual environment.

The scripts and notes of this session are available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-05-deployment/05-deploy.ipynb).
