## 8.8 Adding more dense layers

It is possible to have more than one dense layer in a convolutional neural network. In this way, there is an intermediate processing of the vector representation of images before it goes to the output, and it can be useful to improve the prediction output.

In the neural network layers, we can apply an activation function hat transforms raw scores into probabilities. For the output layers, sigmoid and softmax are the most common choices, while for intermediate layers we can apply other activation functions such as RELU. The RELU activation function output is 0 if x value is lower or equal to 0, and 1 if this value is greater than 0.

In this session, the addition of an extra dense layer did not improve the model performance, so we will not add it into the neural network architecture.

**Classes and methods:**

* `keras.layers.Dense(x, activation='relu')(vectors)` - add an intermediate dense layer to the neural network model with an x size and the relu activation function.
* `watch nvidia-smi` - check he use of the GPU

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).

## 8.9 Regularization and dropout

We want that our neural network models focus on overall shape of the clothes instead of details like logos. One way to approach this problem is hiding parts of the images at each epoch, so at each iteration the model will see a slightly different version of the same image. Formally, this process is called dropout, and it refers to randomly hide or freeze a part of the input of models, particularly some parts of inner layers.

Thus, we regularized the inner layer with dropout, which means that we add some restrictions into our model to avoid overfitting. The drop rate corresponds to the amount of the network that we freeze.

**Classes and methods:**

* `keras.layers.Dropout(x)(inner)` - add regularization with dropout to the inner layer of a neural network with an x dropout rate.

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).

## 8.10 Data augmentation

Data augmentation is the process to generate new data from the available dataset. In the case of images, we can apply some transformations such as horizontal or vertical flipping, clockwise or counter clockwise rotation, horizontal or vertical shifting, pulling only one side of the image or shearing, zoom out or zoom in, adding a black patch on the images, ir a combination of the previous techniques.

To choose the augmentation techniques, we can use our own judgement, look at the dataset and identify variations of data on it, and tune these techniques as a hyperparameter of the model.

We applied augmentation techniques to training dataset, but not for the validation one because we need to compare our model performance with other that did not apply augmentation.

**Classes and methods:**

* `ImageDataGenerator( preprocessing_function=preprocess_input, vertical_flip=True,...)` - load images with keras class with the vertical flip data augmentation technique, and it is possible to add many more transformations of the images.

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).

## 8.11 Training a larger model

In this lesson, we trained a neural network wit larger images - 299x299. Because of the increase of images' size, the model ran slower than the previous one.

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).

## 8.12 Using the model

In this lesson, we loaded the best model trained in the last lesson, evaluated its performance, and used to obtain predictions on new images.

**Classes and methods:**

* `keras.models.load_model('x')` - keras method to load an x model.
* `model.evaluate(x)` - keras method to evaluate a model using an x test dataset.

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).

## 8.13 Summary

* We can use pre-trained models for general image classification.
* Convolutional layers let us turn an image into a vector.
* Dense layers use the vector to make the predictions.
* Instead of training a model from scratch, we can use transfer learning and re-use already trained convolutional layers.
* First, train a small model (150x150) before training a big one (299x299).
* Learning rate - how fast the model trains. Fast learners aren't always best ones..
* We can save the best model using callbacks and checkpointing.
* To avoid overfitting, use dropout and augmentation.

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).
