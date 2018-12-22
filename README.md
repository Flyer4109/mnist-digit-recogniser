# Mnist-digit-recogniser
This is a project for the [digit recognition](https://www.kaggle.com/c/digit-recognizer) kaggle competition that
uses the mnist 28x28 image data set.

This project was run using Python 3.6.6.

Data can be downloaded from
[here](https://www.kaggle.com/c/digit-recognizer/data) and is assumed to be in ../data/.

Dependencies include:
* tensorflow 1.12.0
* keras 2.2.4
* pandas 0.23.4
* numpy 1.15.4
* scikit-learn 0.20.1

A full list of dependencies can be seen at the bottom of this file in case I've missed one out.

There are 3 runnable python scripts available:
* `cv_model.py`
* `save_model.py`
* `submit_model.py`

There are 4 different models available:
* `nn` Neural Network (NN)
* `lstm` Long Short-Term Memory (LSTM) Network
* `cnn` Convolutional Neural Network (CNN)
* `cnn2` CNN

You can see these in `models.py`.

***

## `cv_model.py`
This script uses K-Fold Stratified Cross Validation (CV) to train and evaluate a given model. This script outputs
the score of each fold and then the final score which is the mean of all K folds.

To use this script you must pass 1 or 3 arguments as shown below:

`python cv_model.py <model_type>`

`python cv_model.py <model_type> <n_splits> <verbose>`

`model_type`: string which accepts model codes as mentioned above. `nn`/`lstm`/`cnn`/`cnn2`

`n_splits`: the number of times to split the data set. This is your K value in CV. Default value is 10.
 
`verbose`: 1 for on, 0 for off. Default value is 0.

***

## `save_model.py`
This script finalises a given model by training it using **all** of the data set and then saves it in an h5 file.

To use this script you must pass 2 arguments as shown below:

`python cv_model.py <model_type> <model_name>`

`model_type`: string which accepts model codes as mentioned above. `nn`/`lstm`/`cnn`/`cnn2`

`model_name`: name of the file. For example: neural_network_1 is saved as neural_network_1.h5.


**_INFO_** --- models are saved in ../models/

***

## `submit_model.py`
This script generates a csv file ready for submission by using a given saved model in an h5 file.

To use this script you must pass 2 arguments as shown below:

`python cv_model.py <model_type> <model_file>`

`model_type`: string which accepts model codes as mentioned above. `nn`/`lstm`/`cnn`/`cnn2`

`model_file`: name of the h5 file with the saved model. For example: neural_network_1 will load neural_network_1.h5.

**_INFO_** --- models are read from ../models/

**_INFO_** --- submissions are saved in ../submissions/

***

## Full list of dependencies

* absl-py             0.6.1  
* astor               0.7.1  
* cycler              0.10.0
* gast                0.2.0  
* grpcio              1.16.0 
* h5py                2.8.0  
* Keras               2.2.4  
* Keras-Applications  1.0.6  
* Keras-Preprocessing 1.0.5  
* kiwisolver          1.0.1  
* Markdown            3.0.1  
* numpy               1.15.4 
* pandas              0.23.4 
* pip                 18.1   
* protobuf            3.6.1  
* pyparsing           2.3.0  
* python-dateutil     2.7.5  
* pytz                2018.7 
* PyYAML              3.13   
* scikit-learn        0.20.1 
* scipy               1.1.0  
* setuptools          39.0.1 
* six                 1.11.0 
* tensorboard         1.12.0 
* tensorflow          1.12.0 
* termcolor           1.1.0  
* Werkzeug            0.14.1 
* wheel               0.32.2