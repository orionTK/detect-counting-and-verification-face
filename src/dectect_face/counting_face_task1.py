
import os
import mxnet as mx
import cv2
from utils.face_detector import *
from align_mtcnn.mtcnn_detector import MtcnnDetector
from multiprocessing.pool import ThreadPool
import threading
from multiprocessing import Process
import tensorflow as tf

# from src.dectect_face.utils.face_detector import OpenCVHaarFaceDetector, TensoflowMobilNetSSDFaceDector
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from keras.models import load_model

class CountingFace:

    def __init__(self, type_model):
        self.type_model = type_model

    def Read_File(self):
        with open("../Image/label_count.txt", 'r') as f:
            labels = f.readlines()

        return
    def Check_Face(model, img):
        result = model.predict(img)
        # print(result[0])
        return result[0] == 1
    def Counting(self):
        path_model = os.path.join(os.getcwd(), 'check_face_4.h5')
        model = load_model(path_model)
        true_positive = 0.0
        total = 0.0

        with open("../Image/label_count.txt", 'r') as f:
            labels = f.readlines()
        if self.type_model == "mtcnn":
            mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
            #
            gpu = -1
            if gpu == -1:
                ctx = mx.cpu()
            else:
                ctx = mx.gpu(gpu)
            pre = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)
        else:
            if self.type_model == 'opencv':
                pre = OpenCVHaarFaceDetector(scaleFactor=1.3,
                                                minNeighbors=5,
                                                model_path='models/haarcascade_frontalface_default.xml')
            elif self.type_model == 'mobilenet_ssd':
                pre = TensoflowMobilNetSSDFaceDector(det_threshold=0.3,
                                                        model_path='models/ssd/frozen_inference_graph_face.pb')
            else:
                pre = TensoflowMobilNetSSDFaceDector(det_threshold=0.3,
                                                        model_path='models/ssd/frozen_inference_graph_face.pb')
        # filenames = glob.glob("C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/Count/*.jpg")
        dir = "..\Image\Count";
        # dir_out = "C:/Users/Admin/Desktop/DoAn/src/Image/Count_result/";
        dir_out_fail = "E:/Image_DT/Fail/";
        dir_out_check = "E:/Image_DT/Check/";

        # if not os.path.exists(dir_out):
        #     os.mkdir(dir_out)
        if not os.path.exists(dir_out_fail):
            os.mkdir(dir_out_fail)
        if not os.path.exists(dir_out_check):
            os.mkdir(dir_out_check)
        for labels in labels:
            if len(labels.split(" ")) > 1:
                filename = dir + "\\" +  labels.split(" ")[0]
                # print(filename)
                # print((labels.split(" ")[-1]).split("\n"))
                # TH2
                # number_face = int((labels.split(" ")[-1]).split("\n")[0])
                # th1
                number_face = int((labels.split(" ")[1]).split("\n")[0])
                face_img = cv2.imread(filename)
                # print(face_img)
                # face_img = cv2.resize(face_img, (960,960), interpolation = cv2.INTER_AREA)
                ret = pre.detect_face(face_img)
                if ret is None:
                    continue
                if ret is None:

                    continue

                if self.type_model == "mtcnn":
                    bbox, points = ret
                else:
                    bbox = ret
                if bbox is None:
                    continue
                # print(bbox)



                #(255, 0, 0), 2)

                # check lai tren anh
                #
                total_face_img = 0
                for i, pred in enumerate(bbox):
                    # print("predict: ", pred[0], "\n", pred[1], "\n", pred[3], "\n", pred[2])
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

                    # if pred[1] < 0:
                    #     cv2.rectangle(face_img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255),
                    #                   2)
                    #     cv2.resize(face_img, (224,224))
                    #     cv2.imshow("test", face_img)
                    #     cv2.waitKey(0)
                    #     continue
                    # print(("h ", h))
                    # print(("w ", w))
                    cropped_image = face_img[int(pred[1]):h, int(pred[0]):w, :]

                    if cropped_image == []:
                        cv2.rectangle(face_img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255),
                                      2)
                        cv2.resize(face_img, (224, 224))
                        cv2.imshow("test", face_img)
                        cv2.waitKey(0)
                        continue
                    # cv2.rectangle(face_img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 2)
                    # cropped_image = img.crop((int(pred[0]), int(pred[1]), int(pred[1]), int(pred[3])))
                    # print(cropped_image)

                    cropped_image = cv2.resize(cropped_image, (224, 224))
                    img = cropped_image.reshape(1, 224, 224, 3)
                    # center pixel data
                    img = img.astype('float32')
                    # img = img - [123.68, 116.779, 103.939]
                    # print(img)

                    # predict the class
                    result = model.predict(img)
                    # print(result[0])
                    if result[0] > 0.85:
                        total_face_img += 1
                        # cv2.imshow("check", cropped_image)
                        # cv2.waitKey(0)
                        file_path = '%s_%d.jpg' % ((filename.split("\\")[-1]).split('.')[0], i)
                        #
                        # print('save: ', file_path)
                        file_path_save = os.path.join(dir_out_check, file_path)
                        print('save: ', file_path_save)
                        print(result[0])
                        cv2.imwrite(file_path_save.replace('\\', '/'), cropped_image)

                    else:
                        # cv2.imshow("test", cropped_image)
                        # cv2.waitKey(0)
                        file_path = '%s_%d.jpg' % ((filename.split("\\")[-1]).split('.')[0], i)
                        #
                        print('save: ', file_path)
                        print(result[0])

                        file_path_save = os.path.join(dir_out_fail, file_path)
                        cv2.imwrite(file_path_save.replace('\\', '/'), cropped_image)

                # cv2.rectangle(face_img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 2)
                #     print("predict: ", pred[0], pred[1], pred[2], pred[3])
                #     cv2.rectangle(face_img, (int(pred[0]), int(pred[1])),(int(pred[2]), int(pred[3])), (0, 0, 255), 2)
                #
                #     file_path = '%s_%d.jpg' % ((filename.split("/")[-1]).split('.')[0], i)
                #
                #     print('save: ', file_path)
                #     file_path_save = os.path.join(dir_out, file_path)
                # cv2.putText(face_img, "People: " + str(bbox.shape[0]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 3)
                # cv2.imwrite(file_path_save.replace('\\', '/'), face_img)

                #
                # cv2.imshow(file_path_save.replace('\\', '/'), face_img)
                # cv2.waitKey(0)
                #

                # true_positive += abs(number_face - bbox.shape[0]);
                true_positive += abs(number_face - total_face_img);
                total += number_face
                print(labels.split(" ")[-1].split("\n")[0])
                # print("shape ", bbox.shape[0])
                # print("face ", total_face_img)

                # print("true_positive ", number_face - total_face_img)
                # print("Pedict people: ", number_face)
                # print("Pedict total_face_img: ", total_face_img)
                print(labels)

                print(true_positive)
        return 1 - true_positive / total
if __name__ == '__main__':
    count = CountingFace("mtcnn")

    print(count.Counting())