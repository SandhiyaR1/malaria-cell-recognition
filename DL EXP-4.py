#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn


# In[2]:


pip install matplotlib


# In[1]:


pip install numpy


# In[2]:


pip install pandas


# In[1]:


pip install tensorflow


# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[2]:


import tensorflow as tf

# Configure GPU options for TensorFlow 2.x
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# In[5]:


pip install --upgrade numpy scipy scikit-learn


# In[3]:


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


# In[4]:


pip install os


# In[5]:


my_data_dir = 'dataset/cell_images'
os.listdir(my_data_dir)


# In[6]:


test_path = my_data_dir+'/test/'


# In[7]:


train_path = my_data_dir+'/train/'


# In[8]:


os.listdir(train_path)


# In[9]:


len(os.listdir(train_path+'/uninfected/'))


# In[10]:


len(os.listdir(train_path+'/parasitized/'))


# In[11]:


os.listdir(train_path+'/parasitized')[0]


# In[12]:


para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])


# In[13]:


plt.imshow(para_img)
print('SANDHIYA R 212222230129')


# In[14]:


# Checking the image dimensions
dim1 = []
dim2 = []


# In[15]:


for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


# In[16]:


sns.jointplot(x=dim1,y=dim2)


# In[17]:


image_shape = (130,130,3)


# In[18]:


help(ImageDataGenerator)


# In[19]:


image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


# In[20]:


image_gen.flow_from_directory(train_path)


# In[21]:


image_gen.flow_from_directory(test_path)


# In[24]:


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


# In[25]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[29]:


model.summary()
print("SANDHIYA R \n212222230129 ")


# In[30]:


batch_size = 16
help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')


# In[31]:


train_image_gen.batch_size


# In[32]:


len(train_image_gen.classes)


# In[33]:


train_image_gen.total_batches_seen


# In[34]:


test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


# In[35]:


train_image_gen.class_indices


# In[36]:


results = model.fit(train_image_gen,epochs=3,
                              validation_data=test_image_gen
                             )


# In[37]:


model.save('cell_model.h5')


# In[40]:


losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
print("SANDHIYA R \n212222230129")


# In[41]:


model.metrics_names
model.evaluate(test_image_gen)


# In[42]:


pred_probabilities = model.predict(test_image_gen)


# In[43]:


test_image_gen.classes
predictions = pred_probabilities > 0.5


# In[44]:


print(classification_report(test_image_gen.classes,predictions))
print('SANDHIYA R \n212222230129')


# In[45]:


print(confusion_matrix(test_image_gen.classes,predictions))
print('SANDHIYA R\n212222230129')


# In[ ]:




