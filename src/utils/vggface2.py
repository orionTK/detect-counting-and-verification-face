from numpy import asarray
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
from numpy import expand_dims


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
model1 = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
model2 = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def get_embeddings_vggface2_resnet50(samples):
    # samples = img.astype('float32')
    #
    # samples = expand_dims(samples, axis=0)
    # sử dụng vggface2 nên version = 2
    samples = preprocess_input(samples, version=2)
    yhat = model.predict(samples)
    return yhat

def get_embeddings_vggface2_setnet(samples):
    # samples = img.astype('float32')
    #
    # samples = expand_dims(samples, axis=0)

    # sử dụng vggface2 nên version = 2
    samples = preprocess_input(samples, version=2)

    yhat = model1.predict(samples)
    return yhat

def get_embeddings_vggface2_vgg16(samples):
    # samples = img.astype('float32')
    #
    # samples = expand_dims(samples, axis=0)
    samples = preprocess_input(samples, version=2)

    yhat = model2.predict(samples)
    return yhat