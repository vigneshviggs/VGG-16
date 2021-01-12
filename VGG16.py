# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:17:02 2019

@author: kalpesh
"""

from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns
import numpy as np
import os,cv2

#%%
# Loading the image and Pre-processing

try:
    PATH = os.getcwd()
# Define data path
    data_path = PATH + '\\images'
    data_dir_list = os.listdir(data_path)

    img_rows=224
    img_cols=224
    num_channel=1
#num_epoch=200

# Define the number of classes

    img_data_list=()

    for dataset in data_dir_list:
	     img_list=os.listdir(data_path)
	     print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	     for img in img_list:
		     input_img=cv2.imread(data_path + '\\'+ dataset + '\\'+ img )
		     #input_img_resize=cv2.resize(input_img,(224,224))
		     #img_data_list.append(input_img_resize)

except Exception as e:
    print(str(e))

def _load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img 

def _get_predictions(_model):
    f, ax = plt.subplots(1, 4)
    f.set_size_inches(80, 40)
    #for i in range(4):
        #ax[i].imshow(Image.open(img_data_list[i]).resize((224, 224), Image.ANTIALIAS))
    #plt.show()
    
    f, axes = plt.subplots(1, 4)
    f.set_size_inches(80, 20)
    for i,img_path in enumerate(img_data_list):
        img = _load_image(img_path)
        preds  = decode_predictions(_model.predict(img), top=3)[0]
        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes[i])
        b.tick_params(labelsize=55)
        f.tight_layout()
 
#%%
#Defining the model

_input = Input((224,224,3))        
conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
pool1  = MaxPooling2D((2, 2))(conv2)

conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
pool2  = MaxPooling2D((2, 2))(conv4)

conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
pool3  = MaxPooling2D((2, 2))(conv7)

conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
pool4  = MaxPooling2D((2, 2))(conv10)

conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
pool5  = MaxPooling2D((2, 2))(conv13)

flat   = Flatten()(pool5)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
output = Dense(1000, activation="softmax")(dense2)

vgg16_model  = Model(inputs=_input, outputs=output)

vgg16_model.summary()
#%%
#Prediction

#vgg16_weights = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
vgg16_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5')
_get_predictions(vgg16_model)
print('Predicted:', decode_predictions(preds))

#%%

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

preds = vgg16_model.predict(x)
print('Predicted:', decode_predictions(preds))