
from align_mtcnn.mtcnn_detector import MtcnnDetector
import os
import mxnet as mx
from keras.models import load_model
import tensorflow as tf
import time
import cv2
import glob
path_model = os.path.join(os.getcwd(), 'check_face_4.h5')
# path_model = "E:\Hoc\DoAn\src\dectect_face\build_model.h5"
model = load_model(path_model)

mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
#
gpu = -1
if gpu == -1:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(gpu)
pre = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)

def detetc_face(frame):
    total_face_img = 0
    img = frame.copy()
    # face_img = cv2.imread(img)
    ret = pre.detect_face(img)
    bbox, points = ret
    if bbox is None:
        return total_face_img

    return bbox


all_file_image = glob.glob('../Image/Demo/FaceDetection/*/*.jpg')
dir_out = "E:/Hoc/DoAn/src/Image/Demo/Counted";
for i, path_file in enumerate(all_file_image):
    img_face = cv2.imread(path_file)
    bbox = detetc_face(img_face)

    if bbox is None:
        continue
    total_face_img = 0
    path_file = path_file.replace("\\", '/')
    name_folder = path_file.split('/')[-2]
    name_file = path_file.split('/')[-1]
    for i, pred in enumerate(bbox):
        if pred[1] < 0:
            pred[1] = 0

        if pred[0] < 0:
            pred[0] = 0

        if pred[2] < 0:
            pred[2] = 0

        if pred[3] < 0:
            pred[3] = 0
        h = int(pred[3])
        w = int(pred[2])

        cropped_image = img_face[int(pred[1]):h, int(pred[0]):w, :]

        if cropped_image == []:
            cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255),
                          2)
            cv2.resize(img, (224, 224))
            cv2.imshow("test", img)
            cv2.waitKey(0)
            continue

        cropped_image = cv2.resize(cropped_image, (224, 224), interpolation = cv2.INTER_AREA)
        img = cropped_image.reshape(1, 224, 224, 3)

        img = img.astype('float32')

        # predict the class
        result = model.predict(img)
        print(result[0])
        if result[0] > 0.85:
            total_face_img += 1
            cv2.rectangle(img_face, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 2)

    resize_img = cv2.resize(img_face, (1000, 750))
    cv2.putText(resize_img, "People: " + str(total_face_img), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 3)

    cv2.imshow("image", resize_img)
    cv2.waitKey(500)
    file_path_save = os.path.join(dir_out, name_folder)
    if not os.path.exists(file_path_save):
        os.mkdir(file_path_save)
    file_path = os.path.join(file_path_save, name_file)
    print(result[0])
    cv2.imwrite(file_path, resize_img)






