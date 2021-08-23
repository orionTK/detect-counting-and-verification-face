# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
from tensorflow.keras import layers
from tensorflow import keras

import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob

def build_model():
    PATH = "E:/DataSet_CheckModel_Test/"


    train_dir = os.path.join(PATH, 'train')
    val_dir = os.path.join(PATH, 'val')
    epochs = 10
    batch_size = 15
    IMG_SIZE = (224, 224)

    images_train = glob.glob(os.path.join(train_dir, '*/*.jpg'))
    # print(images_train)
    images_val = glob.glob(os.path.join(val_dir, '*/*.jpg'))
    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )
    train_generator = image_gen_train.flow_from_directory(
        batch_size=2,
        directory=train_dir,
        shuffle=True,
        target_size=IMG_SIZE,
        class_mode='binary'
    )

    image_gen_val = ImageDataGenerator(rescale=1. / 255)

    val_generator = image_gen_val.flow_from_directory(batch_size=2,
                                                      directory=val_dir,
                                                      target_size=IMG_SIZE,
                                                      class_mode='binary')
    conv_base = VGG16(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
    # mark loaded layers as not trainable
    for layer in conv_base.layers:
        layer.trainable = False
    # add new classifier layers
    # for gpu in tf.config.experimental.list_physical_devices('GPU'):
    # 	tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
    model = keras.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation= 'relu'))
    model.add(layers.Dense(1, activation='sigmoid'))


    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit( train_generator,
    steps_per_epoch=int(np.ceil(len(images_train) / float(batch_size))),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=int(np.ceil(len(images_val)/ float(batch_size))))

    model.save('build_model_01.h5')
    summarize_diagnostics(history)

def summarize_diagnostics(history):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title("model accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend(['train'], loc="upper left")
	plt.show()

if __name__ == '__main__':
	build_model()