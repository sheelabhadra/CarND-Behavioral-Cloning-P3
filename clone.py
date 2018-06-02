# importing the required libraries
import os
import csv
import cv2
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D, Cropping2D
from keras.optimizers import Adam

# get the data (Udacity data)
lines = []
with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

# remove the headers
lines = lines[1:]

# split the data into training set and validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# randomly change the brighness of the images to be robust to changes in lighting conditions
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def crop_image(image):
  # the size of the input image is 160x320x3
  # the size of the output image is 75x320x3  
  return image[60:135, :, :]

def resize_image(image):
  # resize the image for input to the network to 100x100x3
  return cv2.resize(image,(100, 100))

def flip_image(image):
  # flip the images horizontally to reduce the bias of the car tending to drive towards the left
  return np.fliplr(image)

def normalize_image(image):
  # to make the training faster
  return (image/255.0 - 0.5) 

correction = 0.25 # steering angle correction for right and left camera images

# Image augmentation using Generator
def data_generator(samples, batch_size=32):
  num_samples = len(samples)
  while True:
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      car_images = []
      steering_angle = []

      for batch_sample in batch_samples:
        steering_center = float(line[3])
        # create adjusted steering measurements for the right and left camera images
        steering_left = steering_center  + correction
        steering_right = steering_center  - correction
        steering = [steering_center, steering_left, steering_right]
        path = 'data/IMG/'
        
        # use the 3 camera images      
        for i in range(3):
            img_bgr = cv2.imread(path + batch_sample[i].split('/')[-1])
            img = img_bgr[:,:,::-1]
            
            # Randomly discard 70% of the center camera images with steering angle less than 0.1 
            if i == 0:
                if abs(steering[i]) <= 0.1:
                    drop_prob = np.random.uniform(0,1)
                    if drop_prob < 0.7:
                        continue

            # horizontally flip the camera images
            flip_prob = np.random.uniform(0,1)
            img_flipped = flip_image(img)

            # change the steering angles for flipped images
            steering_flipped = -1.*steering[i]

            # crop the images
            cropped_img = crop_image(img)
            cropped_flipped_img = crop_image(img_flipped)

            # resize the images
            resized_img = resize_image(cropped_img)
            resized_flipped_img = resize_image(cropped_flipped_img)

            # brightness augmentation
            augmented_img = augment_brightness_camera_images(resized_img)
            augmented_flipped_img = augment_brightness_camera_images(resized_flipped_img)

            # normalize image
            normalized_img = normalize_image(augmented_img)  
            normalized_flipped_img = normalize_image(augmented_flipped_img)  

            # add images and steering angles to data set
            car_images.extend([normalized_img, normalized_flipped_img])
            steering_angle.extend([steering[i], steering_flipped])

      X = np.array(car_images)
      y = np.array(steering_angle)
      yield shuffle(X, y)

# Data augmentation
train_generator = data_generator(train_samples, batch_size=32)
validation_generator = data_generator(validation_samples, batch_size=32)

############################### Comma.ai's model #######################################################
# 3 convolution layers followed by 2 fully connected layers
model = Sequential()
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), input_shape=(100,100,3), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
########################################################################################################
model.summary()

# Compilation
optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='mse')

# Training the model
samples_per_epoch = (9984//32)*32

model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, validation_data=validation_generator,nb_val_samples=2500, nb_epoch=5) 

# Save the model
model.save('model.h5')
print("Model saved")
