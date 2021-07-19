import random
from random import shuffle
from numpy import load
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import glob
from keras.preprocessing.image import ImageDataGenerator,load_img
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
from shutil import copyfile

def create_folder():
    name_home = '../Image/DataSet_CheckModel/'
    subdirs = ['train/', 'val/']
    dataset_home = '../Image/'
    BASE_PATH = '../Image/UTK'
    BASE_PATH1 = '../Image/Fruit'
    all_file_peoples = glob.glob(BASE_PATH + '/*.jpg')
    all_file_people = []
    for i in all_file_peoples:
        all_file_people.append(i.replace('\\', '/'))
    all_file_others = glob.glob(BASE_PATH1 + '/*/*.jpg')
    all_file_other = []
    for i in all_file_others:
        all_file_other.append(i.replace('\\', '/'))
    # for i in all_file_other:
    #     i.replace('\\', '/')
    label_peple = []
    label_other = []

    for i in range(len(all_file_people)):
        label_peple.append(0)
    for i in range(len(all_file_others)):
        label_other.append(1)
    for sub in subdirs:
        labels = ['people/', 'other/']
        for label in labels:
            newdirs = name_home + sub + label
            os.makedirs(newdirs, exist_ok=True)

    # data = pd.read_csv("data_chekcmodel.csv")
    # img_data = data['path'].values
    # label = data['label'].values
    print(len(all_file_other))
    print(len(label_other))

    X_train_people, X_val_people, y_train_people, y_val_people = train_test_split(all_file_people, label_peple, train_size=0.8)
    X_train_other, X_val_other, y_train_other, y_val_other = train_test_split(all_file_other, label_other, train_size=0.8)
    print(X_train_other[25])
    # return X_train, X_val, y_train, y_val

    # count = 0
    # for file in X_train_other:
    #     copyfile(BASE_PATH1 + '/' + file.split('/')[-2] + '/' + file.split('/')[-1], name_home + 'train/other/' + file.split('/')[-1])
    #     count += 1
    #     print(count)

    list(map(lambda file: copyfile(BASE_PATH1 + '/' + file.split('/')[-2] + '/' + file.split('/')[-1], name_home + 'train/other/' + str(random.randint(1 ,100)) + '_' + str(random.randint(1 ,10000)) + '_' + str(random.randint(1 ,1000)) + '_' + file.split('/')[-1]
                        ), X_train_other))
    print(len(X_train_other))
    list(map(lambda file: copyfile( BASE_PATH + '/' + file.split('/')[-1] , name_home + 'train/people/' + file.split('/')[-1]),
             X_train_people))
    list(map(lambda file: copyfile( BASE_PATH + '/' + file.split('/')[-1], name_home + 'val/people/' + file.split('/')[-1] ),
             X_val_people))
    print(len(X_train_people))

    list(map(lambda file: copyfile( BASE_PATH1 + '/' + file.split('/')[-2] + '/' + file.split('/')[-1], name_home + 'val/other/'
                                    + str(random.randint(1, 100)) + '_' + str(random.randint(1 ,1000)) + '_' + str(random.randint(1 ,10000)) + '_' + file.split('/')[-1]), X_val_other))
    print(len(X_val_other))

    print('done!!!')



def load_data():


    print('done!!!')

#
def buil_model():
    # data =  np.load('models/train_data_check.npy', allow_pickle=True)
    FAST_RUN = False
    Image_Width = 224
    Image_Height = 224
    Image_Channels = 3

    # training_data = data
    # print(training_data)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv1', input_shape=(Image_Width, Image_Height , Image_Channels)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_maxpool'))

    model.add(Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_maxpool'))

    model.add(Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv1'))
    model.add( Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_maxpool'))

    model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_maxpool'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2, activation='softmax'))


    model.compile(loss="binary_crossentropy",optimizer='adam', metrics=['accuracy'])
    model.summary()
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, width_shift_range=0.1, height_shift_range=0.1,
                                       horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = train_datagen.flow_from_directory('../Image/DataSet_CheckModel/train/', class_mode='binary',
                                                 batch_size=20, target_size=(224, 224))
    val_it = val_datagen.flow_from_directory('../Image/DataSet_CheckModel/val/', class_mode='binary', batch_size=20,
                                             target_size=(224, 224))
    epochs = 3 if FAST_RUN else 50
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),validation_data = val_it, validation_steps = len(val_it), epochs = 15, verbose = 1)
    # model.fit(X_train, y_train, epochs=50, batch_size= 1, validation_data=(X_val, y_val))

    model.save_weights('models/check_face_model.h5')
if __name__ == '__main__':
    buil_model();

