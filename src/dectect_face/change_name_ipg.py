from keras.models import load_model
import cv2
import glob

all_file_peoples = glob.glob("E:/Image_DT/paper/" + '*.png')
all_file_people = []
for i in all_file_peoples:
    cropped_image = cv2.imread(i)
    name = ((i.replace('\\', '/')).replace('.png', '.jpg')).replace("/paper/","/handadd/")
    cv2.imwrite(name, cropped_image)
