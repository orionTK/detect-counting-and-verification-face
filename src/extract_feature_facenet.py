
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from numpy import asarray
from os import listdir
import timeit
from PIL import Image
from scipy.spatial.distance import cosine
from keras.preprocessing import image
from numpy import savez_compressed
import glob, cv2
import numpy as np
from numpy import savez_compressed, load

from keras.models import load_model

# Lấy model facenet
model = load_model('./models/facenet_keras.h5')
# get the face embedding for one face

def get_face(file_name, required_size=(224, 224)):
    file_name = file_name.replace('\\' or '//', '/')
    name  = file_name.split('/')[-2]
    label = file_name.split('/')[-1]
    print(file_name.split('/')[-2])
    image = cv2.imread(file_name)
    image = Image.fromarray(image)
    img_data = image.resize(required_size)
    face_array = asarray(img_data)
    return face_array, name, label



def get_img(img):
    print(img)
    img = image.load_img(img, target_size=(224, 224))
    img_data = image.img_to_array(img)
    face_array = asarray(img_data)
    return face_array


def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

# load the face dataset
data = load('./exc/dataf.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)
# save arrays to one file in compressed format
savez_compressed('./exc/facenet_embeddings.npz', newTrainX, trainy, newTestX, testy)
print("done!")

if __name__ == "__main__":
    start_time = timeit.default_timer()
    all_file_image = glob.glob('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src//Image/DataIDCroped/*/*.jpg')
    print(all_file_image[2].replace('\\', '/'))

	print(timeit.default_timer() - start_time)
	print('done!')