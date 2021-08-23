from keras.models import load_model
import cv2
import glob
import os
path_model = os.path.join(os.getcwd(), 'check_face_2.h5')
model = load_model(path_model)
face_img = cv2.imread('t5.jpg')

face_img = cv2.resize(face_img, (224, 224))

img = face_img.reshape(1, 224, 224, 3)
# center pixel data
img = img.astype('float32')
result = model.predict(img)
print(result[0])