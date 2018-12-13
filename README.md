

## Project Description

In this project we explore different autonomous driving models in a simulated environment. We begin with a supervised learning method, where the model learns to copy the driving behaviour of a human driver, aka "behavioural cloning". We explore some basic baseline models, such as linear regression and a shallow DNN and then move on to a deeper CNN, known as the NVIDIA model. We then introduce some noise into the model controls in order to simulate unexpected driving disturbances such as a flat tire or a pot hole. Finally we implement some reinforcement learing models with the ultimate end goal of creating a model that can learn to adapt to such potential disturbances.  

### Files included

- model_NVIDIA.py The original NVIDIA model for behavioural cloning. 
- model_NVIDIA_RL.py The modified NVIDIA model for the RL approach.
- model_NVIDIA_RL.py The modified NVIDIA model for the RL approach.
- model_DuelingNetwork.py The modified NVIDIA model for the RL approach.
- drive.py The original script used by the model to communicate with the simulator. 
- drive_RL.py The script used to drive the car with the reinforcement learning algorithm. 
- utils.py The script to provide useful functionalities (i.e. image preprocessing and augumentation)
- model.h5 The model weights.
- environments.yml conda environment (Use TensorFlow without GPU)
- environments-gpu.yml conda environment (Use TensorFlow with GPU)

Note: drive.py is originally from [the Udacity Behavioral Cloning project GitHub](https://github.com/udacity/CarND-Behavioral-Cloning-P3) but it has been modified to control the throttle.

## Quick Start

### Install required python libraries:

You need a [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

```python
# Use TensorFlow without GPU
conda env create -f environment.yml 

# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```

Or you can manually install the required libraries (see the contents of the environemnt*.yml files) using pip.

### Run the pretrained model

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene and press the Autonomous Mode button.  Then, run the model as follows:

```python
python drive.py model.h5
```

### To train the model

You'll need the data folder which contains the training images.

```python
python model_*.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.


## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
