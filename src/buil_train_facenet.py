
import warnings

import src

warnings.simplefilter('ignore')
from os import listdir
from os.path import isdir
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, os, time, random, keras, pickle, gc
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model, Model
from face_verification.facenet import basenet
from face_verification.facenet import train_triplet_generator
from face_verification.facenet import test_triplet_generator
from face_verification.facenet import triplet_loss

#load_img
facesTrain, facesTest = src.load_img('E:/Hoc/BMI/venv/kieu/Data/')


def path_to_list(df):
    paths = list(df)
    count = len(paths)

    return pd.Series([count, paths], index=['count', 'paths'])

vgg2_train_df = facesTrain.apply(path_to_list).reset_index()
vgg2_test_df = facesTest.apply(path_to_list).reset_index()
# define training and test dataset image generator
loops = 10
test_generator = test_triplet_generator(facesTest, batch_size=100, loops=loops, seed=42)

# read test images
size = loops * 500
anchor_imgs = np.zeros((size, 96, 96, 3))
positive_imgs = np.zeros((size, 96, 96, 3))
negative_imgs = np.zeros((size, 96, 96, 3))

for i in range(size // 100):
    images, labels = next(test_generator)
    anchor_imgs[i*100: (i+1)*100] = images[0]
    positive_imgs[i*100: (i+1)*100] = images[1]
    negative_imgs[i*100: (i+1)*100] = images[2]

# load pre-trained model
filepath = './models/facenet-margin-01-final.h5'
model = load_model(filepath=filepath, custom_objects={'tf': tf})

# make predictions
anchor_vec = model.predict(anchor_imgs)
positive_vec = model.predict(positive_imgs)
negative_vec = model.predict(negative_imgs)
print(anchor_vec)

# calcualte the distance
pair_distance = np.sqrt(np.sum(np.square(positive_vec - anchor_vec), axis=1))
diff_distance = np.sqrt(np.sum(np.square(negative_vec - anchor_vec), axis=1))