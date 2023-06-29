# Session-9 Assignment

## Problem Statement

![image](https://github.com/MPGarg/ERA1_Session9/assets/120099863/ca322350-3ed8-4cc2-a64d-bad84d1dce4e)

## Code Details

Code is devided into five files. Four of them are py files & one ipynb file. Link for them are as below:
  * model_cifar10.py [link](model_cifar10.py)
  * model_transpose.py [link](model_transpose.py)
  * train_test.py [link](train_test.py)
  * utility.py [link](utility.py)
  * EVA1_S9.ipynb [link](EVA1_S9.ipynb)
  
All of these py files are imported into EVA1_S9.ipynb after uploading in google drive. model_cifar10.py contains a class by name Net that creates nueral network.

Following is the list of functions that are called for different purpose:
  * load_data: Load transformations for mean and standard deviation calculation
  * set_albumen_params: Albumentation library transformation
  * tl_ts_mod: Apply augmentation to dataset
  * train_test_model: Train & test function
  * display_incorrect_pred: Show incorrect predictions

## Model Design

**Layers for Model:**
 * Convolution Block 1
 * Transition Block 1
 * Convolution Block 2
 * Transition Block 2
 * Convolution Block 3 (It contains Depthwise Separable Convolution & Dilated Convolution)
 * Transition Block 3
 * Convolution Block 4
 * GAP

**Summary of Model:**

![image](https://github.com/MPGarg/ERA1_Session9/assets/120099863/81d2434b-7092-4950-af81-846eb1237161)
![image](https://github.com/MPGarg/ERA1_Session9/assets/120099863/54fc148d-5ccf-45a0-aad4-3e048bf5d514)

**Receptive Field:**

![image](https://user-images.githubusercontent.com/120099863/216605029-6b7b618e-d701-4e47-94d2-e541a8a4a944.png)

**Performance Summary:**
 * Total Trainable Parameters: 182,260
 * Best Training Accuracy: 82.39%
 * Best Test Accuracy: 85.01% (Epoch: 34)

**Training Log (last few steps):**

![image](https://github.com/MPGarg/ERA1_Session9/assets/120099863/ef5a5df6-ab1f-4ede-917f-edbf37598a1e)

**Accuracy & Loss Graph:**

![image](https://github.com/MPGarg/ERA1_Session9/assets/120099863/4a8d896c-f4bf-47af-ad86-016ab33a0a82)

**Overall Prediction Summary:**

![image](https://github.com/MPGarg/ERA1_Session9/assets/120099863/ccbaf617-a554-4150-af84-360b02ede672)

**Incorrect Prediction Samples:**

![image](https://github.com/MPGarg/ERA1_Session9/assets/120099863/3ad8dd6c-59d5-4947-b359-b29eae84f179)

