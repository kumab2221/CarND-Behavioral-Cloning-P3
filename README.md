# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

## The Description
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Submission
---
- [Script used to create and train the model](./create_traning_data.py)
- [Script for drive the car](./drive.py)
- [Trained Keras model](./model.h5)
- [Writeup](./writeup.md)
- [Video recording of a vehicle autonomously traveling on a truck for at least one lap](./video.mp4)
- [Virtual environment with Conda(option)](./Behavioral-Cloning-2.yml)

## Requirement
---
1. Environment for the Behavioral Cloning Project
    - This project requires Python 3.5.2 and the following Python libraries installed:

        - numpy 1.18.5
        - tensorflow 1.3.0
        - socketio 5.0.4
        - flask-socketio 3.0.1
        - python-socketio 3.0.0
        - python-engineio 3.0.0
        - eventlet 0.23.0
        - flask 1.1.2

    - When using the conda environment, please build a virtual environment with the following command.
        ```
        conda env create -f Behavioral-Cloning-2.yml
        ```
1. Simulator
The simulator can be downloaded from [the Udacity of repository](https://github.com/udacity/self-driving-car-sim). In the repository, Downloaded version 2 of Term 1 for Windows.


## Details About Files In This Directory
### `create_traning_data.py`
`create_traning_data.py` creates` model.h5`.There is a place to describe the image path and log file path created by the simulator to generate `model.h5` in the code, so it needs to be modified.
```sh
python create_traning_data.py
```

### `drive.py`
Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.


#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

-a----       2021/01/07     23:36          14855 2021_01_07_14_36_09_105.jpg
-a----       2021/01/07     23:36          14855 2021_01_07_14_36_09_151.jpg
-a----       2021/01/07     23:36          14855 2021_01_07_14_36_09_197.jpg
-a----       2021/01/07     23:36          15210 2021_01_07_14_36_09_251.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

## Licence
---
[MIT](LICENSE)

## Author
---
[kumab2221](https://github.com/kumab2221)