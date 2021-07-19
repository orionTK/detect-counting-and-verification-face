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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def pre_dataset():
	PATH = "../Image/DataSet_CheckModel_Test/"
	train_dir = os.path.join(PATH, 'train')
	validation_dir = os.path.join(PATH, 'validation')
	epochs = 20
	batch_size = 20
	IMG_SIZE = (224, 224)

	train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
	test_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1)

	train_generator = train_datagen.flow_from_directory("../Image/DataSet_CheckModel_Test/train/",
														target_size=(224, 224), batch_size=2, shuffle=True,
														class_mode='binary')
	test_generator = test_datagen.flow_from_directory("../Image/DataSet_CheckModel_Test/test/", target_size=(224, 224),
													  batch_size=2, shuffle=False, class_mode='binary')

def define_model_vgg():
	PATH = "E:/DataSet_CheckModelTest/"

	epochs = 20
	batch_size = 20
	IMG_SIZE = (224, 224)

	train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.2, height_shift_range=0.2,
									   zoom_range=0.2,
									   horizontal_flip=True)
	validation_dir = ImageDataGenerator(
		rescale=1./255
	)

	train_generator = train_datagen.flow_from_directory("E:/DataSet_CheckModel_Test/train/",
														target_size=(224, 224), batch_size=2,
														class_mode='binary')
	val_generator = validation_dir.flow_from_directory("E:/DataSet_CheckModel_Test/val/", target_size=(224, 224),
													  batch_size=2, class_mode='binary')
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
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.summary()

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
	history = model.fit(train_generator, steps_per_epoch=8, epochs=epochs,verbose=1,
					validation_data=val_generator)

	model.save('check_face_4.h5')
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

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
	define_model_vgg()