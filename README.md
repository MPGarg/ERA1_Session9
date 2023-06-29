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
 * Total Trainable Parameters: 79,684
 * Best Training Accuracy: 79.16%
 * Best Test Accuracy: 85.00% (Epoch: 192)

**Training Log (last few steps):**

![image](https://user-images.githubusercontent.com/120099863/216589237-7c887547-83e8-4850-b53f-7598ae29f9a1.png)

**Accuracy & Loss Graph:**

![image](https://user-images.githubusercontent.com/120099863/216590404-6d296358-992d-459b-a1e3-4c94918f7962.png)

**Overall Prediction Summary:**

![image](https://user-images.githubusercontent.com/120099863/216590589-4dcf265f-542a-4aab-ad68-6ecdcda78a25.png)



