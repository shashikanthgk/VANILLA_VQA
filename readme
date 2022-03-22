# Stacked Attention Network with Multi-Layer Feature Fusion For Visual Question Answering

## Introduction
Visual Question Answering(VQA) is a relatively difficult task in computer vision. A lot of different approaches for VQA have been proposed and have been able to achieve good level of accuracy, but the problem of visual question answering is a relatively new field and has a lot of scope for improvement. This paper introduces the concept of 'Multi-layer Feature Fusion' to extract better information from image vectors by taking into account all the layers of a VGG model. 
The implemented models along with their accuracy are as follows
- Vanilla VQA - 57.58%
- Single Attention VQA - 59.8%
- Stacked Attention VQA - 61.6%
- Stacked Attention with Multi-Layer Feature Fusion - Addition Operator, Max Pooling - 61.54%
- Stacked Attention with Multi-Layer Feature Fusion - Addition Operator, Average Pooling - 62.15%
- Stacked Attention with Multi-Layer Feature Fusion - Concatenation Operator, Average Pooling - 63.67%

## Project Structure
Source code is available at https://github.com/shashikanthgk/VANILLA_VQA
- train.py - Contains the code to train the model that is defined in the models.py. It also has options to define hyperparameters while training the model
- models.py - Contains definitions of each models
-- Image Encoder
-- Question Encoder
-- Vanilla VQA 
-- Attention Layer
-- Single Attention VQA
-- Stacked Attention VQA
-- Stacked Attention with Multi-Layer Feature Fusion - Addition Operator
-- Stacked Attention with Multi-Layer Feature Fusion - Concatenation Operator
- plotter.ipynb - Jupyter notebook file used to plot accuracy, loss graph and table in overleaf format
- model_analyzer.py - Contains code to visualize feature vectors obtained from the final layer of CNN
- data_loader.py - Helper class used to load data while training in train.py
- run.sh - Script file used when code is executed on NITK GPU. Defines the number of epochs and batch size
- test.py - Used to generate the accuracy of the model
- utils directory - Contains neccessary utils to create vacabs, resize images and build VQA inputs
- results directory - Contains the results for all the trained models in the result.txt file. Code is being run for 100 epochs and exp1 and exp2 results are noted down
- logs_saved directory - Contains logs for each epoch run. Built in case the model fails midway so we can analyze at which epoch error occured
- 

## How to run
- Download dataset from https://drive.google.com/uc?id=1RHFxESxtFOX1AlK2nTy1mr8jepOV-w_h
- Clone the project repository https://github.com/shashikanthgk/VANILLA_VQA.git on your local machine
- Make a directory using to store datasets and unzip the above file to this directory

```
>mkdir VANILLA_VQA/datasets
>unzip 1MP.zip -d VANILLA_VQA/datasets/
```
- Build vqa inputs using 
```
>python VANILLA_VQA/utils/build_vqa_inputs.py --input_dir='VANILLA_VQA/datasets/1MP' --output_dir='VANILLA_VQA/datasets/1MP'
```
- Train the model using 
```
>python3 VANILLA_VQA/train.py --num_epochs 70 --batch_size 128 --model 'SAN' --save_step 25
```
