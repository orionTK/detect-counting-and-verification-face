import cv2
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras import backend as K
from keras.models import load_model
class Verify_face:
    def __init__(self, path_model = 'C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/models/facenet_keras.h5'):
        self.path_model = path_model
        self.model = load_model(self.path_model)
    def load_model(self):
        model = load_model(self.path_model)
        return model

    def extract_img(self,filename):
        img = cv2.imread(filename)
        if img is None:
            return
        img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img, dtype=K.floatx()) / 255.0
        face_array = asarray(img)
        return face_array

    def get_embedding(self,face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()

        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = self.model.predict(samples)
        return yhat[0]

    def get_distance(self, path1, path2, threshold=5):
            img1 = self.extract_img(path1)
            img2 = self.extract_img(path2)
            if img2 is None or img1 is None:
                return
            emb_img1 = self.get_embedding(img1)
            emb_img2 = self.get_embedding(img2)
            print(emb_img2.shape)
            print('test -----')
            print(emb_img1.shape)
            distance = np.sqrt(np.sum(np.square(emb_img1 - emb_img2)))
            print('distance = ', distance)
            if distance < threshold:
                return  True
            return False



