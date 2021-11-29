# NotesDay43 - Machine Learning Zoomcamp eleventh week

## 9.1 Introduction to Serverless

In this session, we talked about the overview of this chapter. The aim is to deploy the CNN fashion classifier, obtained in the last chapter, using AWS Lambda. The connection with AWS Lambda is created using TensorFlow Lite.

## 9.2 AWS Lambda

AWS Lambda is used to deploy the CNN fashion classifier, obtained in the last chapter. This application allows to deploy ML models without creating any servers (**serverless**). Instead, it only requires to write a function, and this application performs the rest of the work. Thus, AWS Lambda does not require to worry about infrastructure for serving the ML models. Also, for this service you can only pay for requests, or when the service is working.

**Libraries, classes and methods:**

* `lambda_handler(event, context)` - AWS lambda function to manage an event. Usually, the context is None.

## 9.3 TensorFlow Lite

This is a lighter version of TensorFlow (TF) and it is useful because the entire TF is pretty big, its size is more than one gigabyte. We care about the size of TF because in this way the Docker image would be large, so we would need to pay more for storage. Also, the lambda function can have a slow initialization, and it can taje more time to import it, which can create a bigger RAM footprint.

TF Lite is only used for making inferences, not for training models. We used this [model](https://github.com/alexeygrigorev/mlbookcamp-code/releases/download/chapter7-model/xception_v4_large_08_0.894.h5), which is available as part of the MLBookcamp.

**Libraries, classes and methods:**

* `keras.models.load_model('x')` - Keras function to load an x model.
* `load_img('x', target_size=(y, z))` - tensorflow.keras.preprocessing.image function to load an x image with (y,z) size.
* `preprocess_input(x)` - tensorflow.keras.applications.xception function to preprocess an x image represented as a numpy array.
* `tf.lite.TFLiteConverter.from_keras_model(x).convert()` - tensorflow function to convert an x model to tensorflow lite format.
* `Interpreter(model_path='x')allocate_tensors()` - tensorflow.lite function to load an x TF lite model and attach the corresponding weights.
* `Interpreter.set_tensor(input_index, X)` - tensorflow.lite function to initialize the input of the model.
* `Interpreter.invoke()` - tensorflow.lite function to invoke all the computations/layers in the model.
* `Interpreter.get_tensor(output_index)` - tensorflow.lite function to obtain predictions.
* `create_preprocessor('x', target_size=(y, z))` - keras-image-helper function to create a preprocessor for an x model (architecture, i.e. xception) with (y, z) size.

The jupyter notebook of TF lite for AWS Lambda is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/09-serverless/code/tensorflow-model.ipynb).

## 9.4 Preparing the code for Lambda

In this session, we put the TF lite code from the jupyter notebook and convert it into a python script.

**Libraries, classes and methods:**

* `jupyter nbconvert --to script x` - jupyter notebook command line utility to convert a jupyter notebook into a python script.

The python script of the TF lite for AWS Lambda is available [here](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/09-serverless/code/lambda_function.py).
