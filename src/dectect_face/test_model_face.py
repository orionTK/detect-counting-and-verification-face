# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
from tensorflow.keras import layers
from tensorflow import keras

import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import matplotlib as plt
from keras.preprocessing.image import ImageDataGenerator
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
import os

def pre_dataset():
	PATH = "../Image/DataSet_CheckModelTest/"
	train_dir = os.path.join(PATH, 'train')
	validation_dir = os.path.join(PATH, 'validation')
	epochs = 20
	batch_size = 20
	IMG_SIZE = (224, 224)

	train_datagen = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15)
	test_datagen = ImageDataGenerator()

	train_generator = train_datagen.flow_from_directory("../Image/DataSet_CheckModelTest/train/",
														target_size=(224, 224), batch_size=2, shuffle=True,
														class_mode='binary')
	test_generator = test_datagen.flow_from_directory("../Image/DataSet_CheckModelTest/train/", target_size=(224, 224),
													  batch_size=2, shuffle=False, class_mode='binary')

def define_model_vgg():
	PATH = "../Image/DataSet_CheckModelTest/"

	epochs = 20
	batch_size = 20
	IMG_SIZE = (224, 224)

	train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.2, height_shift_range=0.2,
									   zoom_range=0.2,
									   horizontal_flip=True)
	validation_dir = ImageDataGenerator(
		rescale=1./255
	)

	train_generator = train_datagen.flow_from_directory("../Image/DataSet_CheckModelTest/train/",
														target_size=(224, 224), batch_size=2,
														class_mode='binary')
	val_generator = validation_dir.flow_from_directory("../Image/DataSet_CheckModelTest/val/", target_size=(224, 224),
													  batch_size=2, class_mode='binary')
	conv_base = VGG16(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
	# mark loaded layers as not trainable
	for layer in conv_base.layers:
		layer.trainable = False
	# add new classifier layers
	for gpu in tf.config.experimental.list_physical_devices('GPU'):
		tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
	model = keras.Sequential()
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.summary()

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
	history = model.fit(train_generator, steps_per_epoch=8, epochs=epochs,verbose=1,
					validation_data=val_generator)
	summarize_diagnostics(history)
	model.save('check_face.h5')

def summarize_diagnostics(history):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title("model accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend(['train'], loc="upper left")
	plt.show()

def run_test_harness():
# define model
	model = define_model_vgg()
# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
	train_it = datagen.flow_from_directory('../Image/DataSet_CheckModel/train/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory('../Image/DataSet_CheckModel/val/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=0)
	# evaluate model
	# _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	# print('> %.3f' % (acc * 100.0))
	# learning curves

	model.save('check_face.h5')
	print("Done!")
	summarize_diagnostics(history)

if __name__ == '__main__':
	define_model_vgg()