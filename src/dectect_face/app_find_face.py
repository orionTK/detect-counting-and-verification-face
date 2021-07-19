import glob
import os
import sys
import mxnet as mx
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from align_mtcnn.mtcnn_detector import MtcnnDetector
import time
import csv
import os
import glob

from utils.api_find_face import Verify_face


class FindFace:
    def __init__(self, method, image, input_path, output_path):
        self.method = method
        self.image = image
        self.input_path = input_path
        self.output_path = output_path

    def load_method(self):
        gpu = -1
        if gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(gpu)
            # code cũ
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
        mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)
        return mtcnn

    def find_face_Folder(self):
        verify_face = Verify_face()
        count_face_find = 0
        count_face_sample = 0
        flag =True
        #load model
        mtcnn = self.load_method()

        array_emb = []
        if not os.path.exists(self.input_path):
            print(self.input_path + ' not exists!')
            return
        all_name_face = glob.glob(self.input_path + '/*.jpg')
        print(all_name_face)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self.output_path + '_trung'):
            os.mkdir(self.output_path + '_trung')
        for name_img in all_name_face:
            #load image
            name_img = name_img.replace('\\\\', '/').replace('\\', '/')

            print(name_img)
            img = cv2.imread(name_img)
            if img is None:
                print('Not is Image!')
                continue

            #get bouding predict
            ret = mtcnn.detect_face(img)
            print(ret)
            if ret is None:
                print('%s do not find face' % (name_img).split('/')[-1])
                continue
            bbox, points = ret
            if bbox is None:
                print('%s do not find face' % (name_img).split('/')[-1])
                continue
            print(bbox)

            name = (name_img).split('/')[-1]
            for i in range(bbox.shape[0]):
                bbox_ = bbox[i, 0:4]
                # crop
                points_ = points[i, :].reshape((2, 5)).T
                face = mtcnn.preprocess(img, bbox_, points_, image_size='224')
                face_name = '%s_%d.jpg' % (name.split('.')[0], i)
                file_path_save = os.path.join(self.output_path, face_name)
                cv2.imwrite(file_path_save, face)
                # print(array_emb)
                if array_emb != []:
                    for emb in array_emb:
                        print(file_path_save, '------------', emb)
                        check  = verify_face.get_distance(file_path_save, emb)
                        print(check)
                        if check == True:
                            flag = False
                            # print(file_path_save, '------------', emb)
                            path = self.output_path + '_trung'
                            print(os.path.join(path, face_name))
                            cv2.imwrite(os.path.join(path, face_name), face)
                            os.remove(file_path_save)
                            break

                if flag == False:
                    count_face_sample += 1
                    flag = True
                    continue

                array_emb.append(file_path_save)
                count_face_find += 1

        print("Faces = ", count_face_find)
        print("Face sample = ", count_face_sample)

        return count_face_find

    def find_face_img(self):
        # load model
        mtcnn = self.load_method()
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        img = cv2.imread(self.image)
        if img is None:
            print('Not is Image!')
            return

        # get bouding predict
        ret = mtcnn.detect_face(img)
        print(ret)
        if ret is None:
            print('%s do not find face' % (self.image).split('/')[-1])
            return
        bbox, points = ret
        if bbox is None:
            print('%s do not find face' % (self.image).split('/')[-1])
            return
        print(bbox)

        name = (self.image).split('/')[-1]
        for i in range(bbox.shape[0]):
            bbox_ = bbox[i, 0:4]
            print("meo: ", bbox)
            # crop
            points_ = points[i, :].reshape((2, 5)).T
            face = mtcnn.preprocess(img, bbox_, points_, image_size='224')
            face_name = '%s_%d.jpg' % (name.split('.')[0], i)
            file_path_save = os.path.join(self.output_path, face_name)
            cv2.imwrite(file_path_save, face)

        number_face = bbox.shape[0]
        print('So face trong mot anh: ', number_face)
        return number_face


if __name__ == '__main__':
    find_face = FindFace('mtcnn', input_path = '../Image/Test_2', image = '../Image/Test/img_1.jpg', output_path = '../Image/OutTest2')
    # find_face_folder = FindFace('mtcnn', input_path = '../Image/Test', image = '../Image/Test/img_1.jpg', output_path = '../Image/OutTest')
    x = find_face.find_face_Folder()