from keras.models import load_model
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
def load_image(filename):
    cropped_image = cv2.imread(filename)
    cropped_image = cv2.resize(cropped_image, (224, 224))
    img = cropped_image.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    return img

img = load_image('t_01.jpg')
model = load_model('check_face_4.h5')
result = model.predict(img)
print(result[0])


