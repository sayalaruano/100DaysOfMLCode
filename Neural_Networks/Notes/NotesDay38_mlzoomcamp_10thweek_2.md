# NotesDay38 - Machine Learning Zoomcamp tenth week

## 8.4 Convolutional neural networks (CNN)

CNN are neural networks that are mostly used to work with images. In general, these neural networks are composed of three types of layers:

* **Convolutional layers:** consist of some filters or small images, which contain simple shapes. These filters are slid across the images, and we can see how similar the filters are to parts of images being slid. All similarity values between filters an parts of an image are saved on a **feature map**, in which higher values indicate higher similarity. Each filter has its own feature map, which can be treated as images. So, each convolutional layer has some filters, apply them to images, outputs feature maps, and these are passed to the other convolutional layers. Filters of the last convolutional layers become more complex than the ones of starting layers, which is done by combining filters of previous layers. Each convolutional layer can detect progressively more complex features, and the more layers we have it is possible to capture more complex features. The output of convolutional layers are vector representation of images, which have information of features extracted with all the filters from convolutional layers. Consequently, the aim of these layers is to extract vector representation og images.
* **Dense layers:** convert the vector representation of images, obtained from convolutional layers, to predictions. For binary classification tasks, we can apply the sigmoid activation function and obtain the probabilities for the two classes. If we are dealing with multi-class classification tasks, we can apply softmax activation function, which is the generalization of sigmoid function for multiple classes, obtain probabilities to all of the classes, and choose the higher one. These layers are called dense because each element of the input x is connected to each element of the output w, so this is a matrix multiplication between X and W. It is possible to have multiple dense layers.
* **Pooling layers:** convert a feature map into a smaller matrix. This is useful for doing a neural network smaller, and forcing it to have fewer parameters.

## 8.5 Transfer learning

In transfer learning, we have pre-trained neural networks with generic convolutional networks that we do not need to change, and dense layers that are specific to the training dataset, which we should replace. So, we will keep convolutional layers and train new dense layers of the pre-trained models. In this way, the most difficult part can be reused and we transfer this knowledge to a new model.

In multi-class classification tasks, the target variable is represented with one-hot encoding.

In this project, the base-model to extract feature representation of images was Xception, and we added some dense layers to classify the 10 categories of clothes creating our custom model.

To obtain the vector representation of images, we sliced the 3D representation matrix, obtained the average of each of them, and put all these values into a vector. This process is called  as 2D- average pooling. We used a functional style to create the neural network, in which components of the model are used as functions with their proper parameters.

To train the neural networks models, we need some requirements, including an optimizer, the learning rate, and an objective function. The optimizer allows to find the best weights for the model considering a specific objective function to know when the algorithm reaches the optimum.

The raw values of dense lawyer before applying softmax activation function are called logits. It is prefered to maintain these values in dense layers for maintaining numerical stability.

An epoch is an iteration to train a model over the the entire dataset.

**Classes and methods:**

* `ImageDataGenerator(preprocessing_function=x)` - tensorflow.keras.preprocessing.image class to create a generator for reading images using an x preprocessing function.
* `ImageDataGenerator().flow_from_directory('x', target_size=(y,z), batch_size=w)` - ImageDataGenerator method to read images with y height and z width from an x directory, and a w batch size.
* `ImageDataGenerator().class_indices` - ImageDataGenerator method to list categories of the images.
* `Xception(weights='imagenet',include_top=False input_shape=(150, 150, 3))` - tensorflow.keras.applications.xception class to create a Xception pre-trained neural network trained with imagenet dataset, without dense layers, and that receives as input, images with (150, 150, 3) size.
* `Xception().trainable = False` - Xception method to specify that we want to freeze the convolutional layers during training of the model.
* `Xception(keras.Input(shape=(150, 150, 3)), training=False)` - specify the input size of the images for the model
* `keras.Model(inputs, outputs)` -  keras method to put information about inputs and outputs.
* `keras.layers.GlobalAveragePooling2D()(base)` -  add pooling layers to a base model for converting a 3D representation matrix to a vector representation of images.
* `keras.layers.Dense(10)(vectors)` -  keras method to create dense layers for transforming vector representation of images into predictions.
* `keras.optimizers.Adam(learning_rate=x)` -  keras method to create an optimizer class with an x learning rate.
* `keras.losses.CategoricalCrossentropy(from_logits=True)` -  keras method to create a CategoricalCrossentropy class for evaluating if an optimizer reaches the optimum for a multi-class classification problem. It is recommended to change from_logits parameter to True because in this way the calculations are numerically stable.
* `model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])` - Xception method to compile the optimizer, loss function and performance metrics a model before training it.
* `model.fit(train_ds, epochs=10, validation_data=val_ds)` - Xception method to compile the optimizer, loss function and performance metrics a model before training it.

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).

## 8.6 Adjusting the learning rate

THe learning rate is the speed in which a model can learn. If this value is high, the model learning is superficial and is prone to overfitting, while if this value is low the learning process is slow and the model tend to underfitting. So, it is important to find the right balance for tuning this parameter.

To fin-tune the learning rate, we trained models with different values of this parameter, and plotted the performance of these models in training and validation datasets using learning curves.

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).

## 8.7 Checkpointing

Checkpointing is a way of saving a model after each iteration of the training, or when a model reach certain conditions. In keras, it is possible to implement checkpointing using callbacks.

**Classes and methods:**

* `keras.callbacks.ModelCheckpoint('xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5', save_best_only=True, monitor='val_accuracy', mode='max')`- keras class to save the best models of all epochs during the training, maximizing the performance metric.

The Jupyter notebook with code for this session is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/08-deep-learning/notebook.ipynb).
