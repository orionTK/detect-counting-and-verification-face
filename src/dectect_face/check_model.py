from keras.models import load_model
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
def load_image(filename):
    face_img = load_img(filename, target_size=(224, 224))
    face_img = img_to_array(face_img)
    face_img.reshape(1, 224, 224, 3)
    # face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
    img = face_img.astype('float32')
    # img = img - [123.68, 116.779, 103.939]
    return img

img = load_image('t1.jpg')
model = load_model('final_check_people_model.h5')
result = model.predict(img)
print(result[0])


