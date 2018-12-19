# Mnist-digit-recogniser
This is a project for the [digit recognition](https://www.kaggle.com/c/digit-recognizer) kaggle competition that
uses the mnist 28x28 image data set. This project was run using Python 3.6.6.

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

## `cv_model.py`
This script uses K-Fold Stratified Cross Validation (CV) to train and evaluate a given model. This script outputs
the score of each fold and then the final score which is the mean of all K folds.

To use this script you must pass 1 or 3 arguments as shown below:

`python cv_model.py <model_type>`

`python cv_model.py <model_type> <n_splits> <verbose>`

***

`model_type`: string which accepts model codes as mentioned above. `nn`/`lstm`/`cnn`/`cnn2`

`n_splits`: the number of times to split the data set. This is your K value in CV. Default value is 10.
 
`verbose`: 1 for on, 0 for off. Default value is 0.

## `save_model.py`
This script finalises a given model by training it using **all** of the data set and then saves it in an h5 file.

To use this script you must pass 2 arguments as shown below:

`python cv_model.py <model_type> <model_name>`

***

`model_type`: string which accepts model codes as mentioned above. `nn`/`lstm`/`cnn`/`cnn2`

`model_name`: name of the file. For example: neural_network_1 is saved as neural_network_1.h5.


**_INFO_** --- models are saved in ../models/

## `submit_model.py`
This script generates a csv file ready for submission by using a given saved model in an h5 file.

To use this script you must pass 2 arguments as shown below:

`python cv_model.py <model_type> <model_file>`

***

`model_type`: string which accepts model codes as mentioned above. `nn`/`lstm`/`cnn`/`cnn2`

`model_file`: name of the h5 file with the saved model. For example: neural_network_1 will load neural_network_1.h5.

**_INFO_** --- models are read from ../models/

**_INFO_** --- submissions are saved in ../submissions/