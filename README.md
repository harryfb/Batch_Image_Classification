
# Batch Image Classification

This program was written as part of a final year university project to build an autonomous land survey vehicle. As the vehicle surveyed a section of park, it captured images of obstacles discovered in its path. The program was designed to run on an Amazon AWS instance, batch processing said images sent to it from the vehicle at the end of the survey. The capture coordinates & corresponding obstacle classifications were then sent to a second server, running an arcGIS script, that drew the correct symbols onto the final map. As the camera on the rover was triggered via restrictive range finding sensors (2D cross sectional LIDAR & ultrasonic), it occasionally incorrectly triggered upon approaching steep grass verges. The program includes a slope detection function that averages the colour in horizontal bands across the image and separates them from true obstacles, based on the resulting RGB colour value.


## Getting Started

The following instructions will aid you in getting the project up and running on your local system. 


### Prerequisites

Before running the code you will need to install the following packages. These instructions are for Mac users and Windows users should refer to the installation guides in the package documentation.  


#### Python 2.7
Python can be installed using the homebrew package manager. If you do not already have this, it can be obtained by following the instructions [here](https://brew.sh/).

Use the following command to begin the installation process.
```
    brew install python@2
```

Full step-by-step instructions on installing python can be found [here](https://medium.com/@yangnana11/installing-python-2-on-mac-os-x-d0f1c9c4d808).


#### Watchdog
An observer is used to run the program upon modification of a file in a specified directory. The package is installed using the command

```
    pip install watchdog
```


#### OpenCV
OpenCV contains many powerful functions for image manipulation. It is used primarily for the detection of slopes in the program.


#### Tensorflow
Tensorflow provides the image classification functionality and is an essential component of the program. It is used to both generate models and run inferences on them.

```
    pip install "tensorflow>=1.7.0"
    pip install tensorflow-hub
```


## Running
To run the program, use the following command.

````
    python image_classify.py
````

It assumes that the "img_tags" CSV file and the input images are located in the "inputs" folder. The graph should be saved in the project directory and the results are saved in the "output" directory.


### CNN Retraining
The current graph was created to identify bins, benches and trees. With some edits, the program could be easily modified in order classify other types of objects. A new graph can be quickly created by following the steps in the [Image Retraining](https://www.tensorflow.org/hub/tutorials/image_retraining) Tensorflow tutorial.


## Authors
* ****Harry F Bullough**** - Initial development (March 2017). Ported to TensorFlow Release 1.13.0 (April 2019) - [harryfb](https://github.com/harryfb)


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


