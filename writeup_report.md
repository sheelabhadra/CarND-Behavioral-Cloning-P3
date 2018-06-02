# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/sample_images.png
[image2]: ./examples/steering_angles.png 
[image3]: ./examples/histogram.png
[image4]: ./examples/flipped_simage.png "Flipped image"
[image5]: ./examples/brightness_aug.png
[image6]: ./examples/cropped_image.png
[image7]: ./examples/resized_image.png
[image8]: ./examples/cnn_arch.png
[image9]: ./examples/sample_image.png "Original image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Data

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 convolution layers and 2 fully connected layers. This model architecture has been used by a [comma.ai](https://comma.ai/) for driving a self-driving car in a simulator (clone.py lines 120-132).

The model includes ELU layers to introduce nonlinearity. (clone.py lines 122, 124, 128, 131)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in the fully connected layers in order to reduce overfitting (clone.py lines 127, 130). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 143). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 137). The initial learning rate was set as 0.0001.

#### 4. Appropriate training data

I used the [training data]() provided by Udacity to train the convolution neural network. I felt that the data would be sufficient because it had 8036 samples. I added more images to the dataset by using a few data augmentation techiques. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
---
The overall strategy for deriving a model architecture was to start with a very simple convolution neural network. This is because the most important cue for determing the steering angle was to determine the curvature of the road. At the same time it was also important for the network to distinguish between road and markings, gravels etc. along the sides of the road. So, I used color images to help the network learn these details.   

My first step was to use a convolution neural network model similar to the LeNet-5. I thought this model might be a good starting point because the most important cue in the images to control the steering angle is the curvature of the road which can be easily detected by using simple edge detectors.    

I used [ELU()](https://arxiv.org/abs/1511.07289) as the activation layer because they are found to work better than RELU in ImageNet challenges.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found implied that the model was underfitting due to high training and validation mean squared loss. Also the car's performance on the track in the autonomous driving mode was not good as it was falling off the track at the corners.

I next used the a relatively complex convolution neural network model used in a [Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper. Although I found that the network this led to a reduction in training and validation mean squared loss, the car didn't perform very well due to overfitting.

To combat the overfitting, I modified the model to add a dropout layers after every layer. After making a few tweaks to the dropout probabilities, the car was able complete the track in autonomous driving mode.

This led me to settle down at an architecture that had a lower complexity than Nvidia's architecture. Comma.ai's convolutional neural network model as described in this [Github repo](https://github.com/commaai/research/blob/master/train_steering_model.py) worked perfectly well on the track. It contains 3 convolution layers follwed by 2 fully connected layers with dropout layers in the fully connected layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.


| Layer           | Kernel Size | Number of filters | Stride    |      Description               | Padding  |
|:---------------:|:-----------:|:-----------------:|:---------:|:------------------------------:|:--------:|
| Input           |     -       |       -           |       	|  100x100x3 pre-processed image |    -     |   
| Convolution     |     8x8     |	    16          |   4x4     |  outputs 25x25x16 	         | Same     | 
| ELU             |		-       | 		-			|    -	    |  activation function           |    -     |
| Convolution     |     5x5     |	    32          |   2x2     |  outputs 13x13x32              | Same     | 
| ELU             |		-       | 		-			|    -	    |  activation function           |    -     |
| Convolution     |	    5x5     |       64          |   2x2     |  outputs 7x7x64              	 |	Same	|
| Flattening      |	    -       |     -             |      -    |  outputs 3136 	             |	   -	|
| Fully Connected |	    -       |     -             |      -    |  3136 input, 512 output 	     |	   -	|
| ELU             |	    -       |     -             |      -    |  activation function 	         |	   -	|
| Dropout         |	    -       |     -             |      -    |  drop probability (0.2)  	     |	   -	|
| Fully Connected |	    -       |     -             |      -    |  512 input, 1 output 	         |	   -	|
| ELU             |	    -       |     -             |      -    |  Activation function 	         |	   -	|
| Dropout         |	    -       |     -             |      -    |  drop probability (0.5)  	     |	   -	|
| Output          |	    -       |     -             |      -    |  steering angle    	         |	   -	|

Here is a visualization of the architecture.

![alt text][image8]

#### 3. Creation of the Training Set & Training Process

I used the dataset provided by Udacity to train the CNN. Below are a few sample images from the dataset.

![alt text][image1]

Since the track is in the form of a circle and the dataset contained images collected by driving car in clockwise direction only, there were more number of right steering angles as can be seen from the histogram of steering angles. 

![alt text][image3]

There are a lot of steering angles close to 0 which tend to make the network biased towards giving values close to 0 steering angle in the output. To remove this bias, I randomly discarded 70% of the data samples that had steering angles between [-0.1, 0.1].

To further improve the distribution of steering angles I used a few data augmentation techniques which have been explained next.

#### Data Augmentation  
To balance the data and to add more meaningful data to my training data without manually driving the car around the tracks, I used a few data augmentation techniques as discussed in the Nvidia paper and this [blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9).

**Using the right and left camera images**  
There are 3 cameras attached at the front of the car in the center, left, and right. I used the left and right camera images for recovery i.e. I added a small angle of 0.25 to the steering angle for left camera and subtracted a small angle of 0.25 from the right camera. The main idea being the left camera has to move right, and right camera has to move left to get to the center. So, basically I treated the left and right camera images as center images by adjusting the steering angles. This led to a 3 fold increase in training data.  

This also works as the recovery data which helps the vehicle recover from the left side and right sides of the road back to the center. This helps the vehicle to learn to return to the center of the track everytime it sways away from it. For example, here is an example showing the center, left, and right images with the modified steering angles.

![alt text][image2]

**Flipping the images and reversing the steering angles**   
To remove the bias of the network to move towards the right as it was driven in the clockwise direction along the track during training, I flipped all the images so that they would resemble driving along the track in the anti-clockwise direction. This helped me balance the steering angle distribution to be more even and symmetric about the 0 steering angle.  
For example, here is an image that has then been flipped:

![alt text][image9]
![alt text][image4]

**Brightness augmentation**  
I also randomly changed the brightness of the images in the datatset to simulate driving in different lighting conditions.

For example, here are a few images after brightness augmentation.

![alt text][image5]

After the data augmentation process, I had about 30000 data samples; 4 times the data in the original dataset.

#### Data Pre-processing
Since, the scenery contained in the images are not necessary to predict the steering angle I removed 60 pixels from the upper half portion of the images.  
Also, the hood of the car was visible at the bottom of the image which I removed 25 pixels from the bottom as well.

Here's how a sample image looks like after cropping.

![alt text][image6]

Further, I resized the images to a size of 100x100 to increase training and processing speed. After images, the images contain sufficient information to identify the curvature of the roads.  

Here's how a sample image which is being fed to the CNN looks like after data pre-processing.

![alt text][image7]

I performed all the pre-processing steps before feeding them to the model to avoid repeating the pre-processing steps in every epoch of training.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as beyond that the training loss and the validation loss remained almost same and the performance of the vehicle on the track did not seem to improve. I used the adam optimizer as it doesn't require the learning rate to be adjusted manually.

**Training method**  
Optimizer: Adam Optimizer
Starting learning rate: 0.0001
No. of epochs: 5
Images generated per epoch: 9984 images generated on the fly
Validation Set: 2500 images, generated on the fly
Keras' fit\_generator method was used to train images generated by the generator

**Video**  
[Click here](https://www.youtube.com/watch?v=5QzUQaIAk2Q&t=20s) to watch the video of the vehicle running in autonomous mode on Track 1.
