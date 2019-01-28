# CNN IoT Integration
The purpose of this project is to integrate an IoT device that I designed (called a Blip) with a CNN.   This will use a Convolution Neural Net (CNN) implemented using Keras with TensorFlow backend to look at a set of Blips, determine the order and display the order on the screen.  A Blip is an IoT device that I created that uses an Electric Imp (https://www.electricimp.com/) chip.  It has LED’s an eInk display and a 3D printed case and a membrain keypad.

You can see it working in the GIF below.  
* First the program displays unique images on each blip
* The program take a picture and passes it to the CNN
* The CNN determines the order of the blips
* The system tells each blip what the order is and the blip displays it

![alt text](https://github.com/enyeartc/blipgit/blob/master/screencap.gif)

## Process
Each blip can display a unique image and display a blip sequence number. The CNN will process an image and determine which of 6 states is in the image.  For this example the unique images are Turtle(T), YingYang(Y), or House(H).  So the 6 states could be TYH, THY, HTY, HYT, YHT, or YTH.   These are the 6 states that the CNN will predict  

The process (in camera_predict_pub.py):
* Run camera_predict_pub.py 
* Program starts camera, user points camera to Blips 
* Program tells blips to display unique image 
* Blips display images
* CNN processes the image
* CNN predicts one of 6 states
* Program sends Blips which sequence to display


## Setup 

* First I created 500 images of each possible state, these images are black and white and 100x200 in size (camera.py)
* I then trained a CNN on these images (blip_cnn_pub.py)
* Code for the Blips will not be displayed in this repo
