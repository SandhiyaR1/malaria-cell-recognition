# Deep Neural Network for Malaria Infected Cell Recognition
#### DATE:
## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
To develop a deep neural network model to classify microscopic cell images as either malaria-infected or uninfected. This is essential for aiding in the rapid and accurate diagnosis of malaria. The dataset includes a structured directory with images of both infected and uninfected cells. To enhance model accuracy, various techniques like hyperparameter tuning, model architecture adjustments, or transfer learning can be applied. Using an ImageDataGenerator, image data is loaded from the dataset directory, with configurations for data augmentation. The model's effectiveness is evaluated through metrics, a confusion matrix, and a plot of training vs. validation loss over epochs
## Neural Network Model

![image](https://github.com/user-attachments/assets/7e18063c-4995-496b-8dd4-501a719d65fa)

## DESIGN STEPS

### STEP 1:
Import Libraries: Load necessary libraries for data processing, visualization, and building the neural network.
### STEP 2:
Configure GPU: Set up TensorFlow to allow dynamic GPU memory allocation for efficient computation.
### STEP 3:
Define Dataset Paths: Set directory paths for training and testing data to organize the data flow.
### STEP 4:
 Load Sample Images: Visualize example images from both infected and uninfected classes to better understand the dataset.
### STEP 5:
 Explore Data Dimensions: Check image dimensions and visualize distribution patterns to assess consistency.
### STEP 6:
Set Image Shape: Define a consistent image shape to standardize inputs for the model.
### STEP 7:
 Compile Model: Set the loss function, optimizer, and evaluation metrics to prepare the model for training.
### STEP 8:
Train the Model: Train the model over a specified number of epochs using the training and validation datasets.
### STEP 9: 
Evaluate and Visualize: Assess model performance, generate metrics, and plot training and validation loss to monitor progress.
 
## PROGRAM

#### Name: SANDHIYA R

#### Register Number:212222230129
```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
my_data_dir = 'dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[0]
para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])
plt.imshow(para_img)
print('SANDHIYA R 212222230129')
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
model = models.Sequential()
model.add(keras.Input(shape=(image_shape)))
# Add convolutional layers
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# Flatten the layer
model.add(layers.Flatten())
# Add a dense layer
model.add(layers.Dense(128, activation='relu'))
# Output layer
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
print("SANDHIYA R \n212222230129 ")
batch_size = 16
help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=3,
                              validation_data=test_image_gen
                             )
model.save('cell_model.h5')
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
print("SANDHIYA R \n212222230129")
model.metrics_names
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
print('SANDHIYA R \n212222230129')
print(confusion_matrix(test_image_gen.classes,predictions))
print('SANDHIYA R\n212222230129')
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/49a35063-4012-46ca-ad6a-ee0cdc8141d9)

### Classification Report

![image](https://github.com/user-attachments/assets/4f42aafd-519d-41f6-add5-24ca92b35831)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/aa5c6752-210c-4527-9ece-4ade189409ec)

### New Sample Data Prediction

![WhatsApp Image 2024-11-21 at 18 06 35_e9418db5](https://github.com/user-attachments/assets/8d52a329-555f-4496-bd6e-f2552f1f93e5)

## RESULT
Thus a deep neural network for Malaria infected cell recognition and to analyze the performance is created using tensorflow.
