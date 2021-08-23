
from align_mtcnn.mtcnn_detector import MtcnnDetector
import os
import mxnet as mx
from keras.models import load_model
import tensorflow as tf
import time
import cv2
# path_model = os.path.join(os.getcwd(), 'check_face_4.h5')
path_model = "E:\Hoc\DoAn\src\dectect_face\check_face_4.h5"
model = load_model(path_model)

mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
#
gpu = -1
if gpu == -1:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(gpu)
pre = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)

def count_face_imgage(frame):
    total_face_img = 0
    img = frame.copy()
    # face_img = cv2.imread(img)
    ret = pre.detect_face(img)
    bbox, points = ret
    if bbox is None:
        return total_face_img

    return bbox

cap = cv2.VideoCapture('../Image/Demo/demo_count.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
initial_target = int(45 * fps) + 10
final_target = int(49 * fps) + 10
i = 0
frames = []
while(True):
    ret, frame = cap.read()
    i +=1
    if i in range(initial_target, final_target, 1):
        frames.append(frame[:,:,::-1])
    if i == final_target:
        break


font = cv2.FONT_HERSHEY_SIMPLEX

detections = []
count_face = []
for i, frame in enumerate(frames):
    total_face_img = 0
    with tf.Graph().as_default():
       detect = count_face_imgage(frame)
    detections.append(detect)
    img = frame.copy()
    for j, pred in enumerate(detect):
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

        cropped_image = img[int(pred[1]):h, int(pred[0]):w, :]

        if cropped_image == []:
            # cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255),
            #               2)
            # cv2.resize(img, (224, 224))
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            continue

        cropped_image = cv2.resize(cropped_image, (224, 224))
        box_co = cropped_image.reshape(1, 224, 224, 3)
        box_co = box_co.astype('float32')
        cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 2)
        result = model.predict(box_co)
        if result[0] > 0.85:
            total_face_img += 1
            cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 2)
    count_face.append(total_face_img)
for i, frame in enumerate(frames):
    cv2.putText(img, 'Incremental count : %d' % count_face[i], (50, 50), font, 1.5, (0, 255, 0), 3)
    cv2.imshow('Video', img)

    # cv2.imwrite('./output_video/frames_%00d.png' % i, img[:,:,::-1])
    k = cv2.waitKey(50) & 0xff
    # if k == 27:
    #     break
cap.release()








