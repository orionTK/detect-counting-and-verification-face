import os
import sys
import mxnet as mx
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from utils.face_detector import *
from align_mtcnn.mtcnn_detector import MtcnnDetector
import time
import math
import csv
import glob
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../Image/Demo/FaceDetection', help='the dir your dataset of face which need to crop')
    parser.add_argument('--output_path', type=str, default='../Image/Demo/FaceDetectionProcess', help='the dir the cropped faces of your dataset where to save')
    # parser.add_argument('--face-num', '-face_num', type=int, default=1, help='the max faces to crop in each image')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id， when the id == -1, use cpu')
    parser.add_argument('--face_size', type=str, default='256', help='the size of the face to save, the size x%2==0, and width equal height')
    args = parser.parse_args()
    return args


def get_iou(boxA, boxB):
    bb1 = dict()
    bb1['x1'] = boxA[0]
    bb1['y1'] = boxA[1]
    bb1['x2'] = boxA[2]
    bb1['y2'] = boxA[3]

    bb2 = dict()
    bb2['x1'] = boxB[0]
    bb2['y1'] = boxB[1]
    bb2['x2'] = boxB[2]
    bb2['y2'] = boxB[3]


    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes area
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


# lay label da dan cho bo du lieu DataIDFull
def get_data_DataIDFull():
    i = 0
    bb_gt_collection = dict()
    list_file_none = []

    all_file_image = glob.glob('../Image/Demo/FaceDetection/*/*.jpg')
    # print(all_file_image)
    if all_file_image != []:
        for file in all_file_image:
            file = file.replace('\\', '/')
            name_img = (file.split('/')[-1])[0:(len(file.split('/')[-1]) - 4)]
            # print(name_img)
            if not os.path.exists(os.path.join('E:/Hoc/DoAn/src/Image', 'label',
                                               '%s.txt' % (name_img))):
                list_file_none.append(file)
                continue
            with open(os.path.join('E:/Hoc/DoAn/src/Image', 'label',
                                   '%s.txt' % (name_img)), 'r') as f:
                lines = f.readlines()
            image_path = file
            bb_gt_collection[image_path] = []
            for line in lines:
                i += 1
                line = line.split('\n')[0]


                if cv2.imread(image_path) is None:
                    continue

                # print(cv2.imread(image_path))

                line_components = line.split(' ')
                # print('....................', line)
                if len(line_components) > 1:
                    # print(line_components[1])
                    img = cv2.imread(image_path)
                    # print(image_path)
                    img_h, img_w, d = img.shape
                    x1 = float(line_components[1].replace(',', '.')) * img_w
                    y1 = float(line_components[2].replace(',', '.')) * img_h
                    w = float(line_components[3].replace(',', '.')) * img_w
                    h = float(line_components[4].replace(',', '.')) * img_h

                    if w > 15 and h > 15:
                        # print("test: ", image_path, line_components)

                        bb_gt_collection[image_path].append(
                            np.array([x1 - w / 2, y1 - h / 2, x1 + w / 2, y1 + h / 2]))
                    # print('-----', bb_gt_collection)


    np.save('file_name.npy', list_file_none)
    return bb_gt_collection

def calculate_metric(args, bb_gt_collection, type_model):
    font = cv2.FONT_HERSHEY_SIMPLEX
    output_dir = args.output_path
    total_data = len(bb_gt_collection.keys())
    # print('===============================================', bb_gt_collection, "+++++++++++++++++++++++++++++++++++++++")
    data_total_iou = 0
    data_total_precision = 0
    data_total_inference_time = 0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # print("meo", bb_gt_collection)
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)
    detect = ''
    mtcnn_path = os.path.join(os.path.dirname(__file__), '../demo/align_mtcnn/mtcnn-model')
    mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)
    detect = mtcnn
    if type_model == 'opencv':
        detect = OpenCVHaarFaceDetector(scaleFactor=1.3,
                 minNeighbors=5,
                 model_path='models/haarcascade_frontalface_default.xml')
    elif type_model == 'mobilenet_ssd':
        detect = TensoflowMobilNetSSDFaceDector(det_threshold=0.3,
                 model_path='models/ssd/frozen_inference_graph_face.pb')


    count_no_find_face = 0
    count_crop_images = 0

    total_data = len(bb_gt_collection.keys())
    for i, key in tqdm(enumerate(bb_gt_collection), total=total_data):

        # print("key: ", (key.split('/')[0]).split('\\')[-1])
        out_path = key.split('/')[-2]
        # print('out_path', out_path)
        # if out_path == 'C:':
        #     out_path = (key.split('/')[-2])
        #     print('2', out_path)
        # print('path: ', out_path)
        # if out_path != 'C':
        filename = os.path.join(output_dir, out_path)
        # print('name: ',filename)
        if not os.path.exists(filename):
            os.mkdir(filename)
        # print('key: ', key)
        # gốc
        image_data = cv2.imread(key)
        # print("image_data ==============", image_data)
        # đã gán
        face_bbs_gt = np.array(bb_gt_collection[key])
        total_gt_face = len(face_bbs_gt)

        start_time = time.time()
        ret = detect.detect_face(image_data)
        inf_time = time.time() - start_time
        data_total_inference_time += inf_time
        if ret is None:
            continue
        if type_model == 'mtcnn':
            face_pred, points = ret
        else:
            face_pred = ret

        ### Calc average IOU, Precision, and Average inferencing time ####
        total_iou = 0
        tp = 0
        pred_dict = dict()
        # print("shape:" ,face_bbs_gt)
        for gt in face_bbs_gt:
            max_iou_per_gt = 0
            # red for label bounding box
            cv2.rectangle(image_data, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (255, 0, 0), 2)
            for i, pred in enumerate(face_pred):
                # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++", key)
                # print("predict: ", pred[0], pred[1], pred[2], pred[3])
                # print("gt: ", gt)
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                #     red for predict bounding box
                cv2.rectangle(image_data, (int(pred[0]), int(pred[1])),(int(pred[2]), int(pred[3])), (0, 0, 255), 2)

                #
                # file_path = '%s_%d.jpg' % (name_img.split('.')[0], i)
                # print('save: ', file_path)
                # file_path_save = os.path.join(filename, file_path)
                #
                # print('file: ', file_path_save.replace('\\', '/'))
                # cv2.imwrite(file_path_save.replace('\\', '/'), face)
                # cv2.imwrite(file_path_save.replace('\\', '/'), face)
                iou = get_iou(gt, pred)
                # cv2.imshow("meo", image_data)
                # cv2.imwrite("meo1.jpg", image_data)
                # cv2.waitKey(0)

                # print('iou', iou)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt
        # save a image
        name_img = key.split('/')[-1]
        file_path = '%s_%d.jpg' % (name_img.split('.')[0], i)
        # print('save: ', file_path)
        # cv2.resize(image_data, (300, 300))

        file_path_save = os.path.join(filename, file_path)
        cv2.imshow("image", image_data)
        # cv2.putText(image_data, 'Red: Label ', (50, 50), font, 1.5, (0, 255, 0), 3)
        # cv2.putText(image_data, 'Green: Predict ', (50, 60), font, 1.5, (0, 255, 0), 3)

        cv2.waitKey(500)
        # print('file: ', file_path_save.replace('\\', '/'))
        cv2.imwrite(file_path_save.replace('\\', '/'), image_data)


        if total_gt_face != 0:
            if len(pred_dict.keys()) > 0:
                for i in pred_dict:
                    # print('pred', pred_dict[i])
                    if pred_dict[i] >= 0.5:
                        tp += 1
                precision = float(tp) / float(total_gt_face)
                # print(precision)

            else:
                precision = 0

            image_average_iou = total_iou / total_gt_face
            image_average_precision = precision

            data_total_iou += image_average_iou
            data_total_precision += image_average_precision

        ## crop su dung de crop and align
        # if type_model == "mtcnn":
        #     for i in range(face_pred.shape[0]):
        #         bbox_ = face_pred[i, 0:4]
        #         # print("meo: ", face_pred)
        #         # cv2.rectangle(image_data, (int(pred[0]), int(pred[1])),(int(pred[2]), int(pred[3])), (225, 0, 0), 2)
        #
        #         # crop
        #         points_ = points[i, :].reshape((2, 5)).T
        #         face = mtcnn.preprocess(image_data, bbox_, points_, image_size=args.face_size)
        #         name_img = key.split('/')[-1]
        #
        #         file_path = '%s_%d.jpg' % (name_img.split('.')[0], i)
        #         print('save: ', file_path)
        #         file_path_save = os.path.join(filename, file_path)
        #
        #         print('file: ', file_path_save.replace('\\', '/'))
        #         cv2.imwrite(file_path_save.replace('\\', '/'), face)


    result = dict()
    result['average_iou'] = float(data_total_iou) / float(total_data)
    print(data_total_precision)
    print(total_data)
    result['mean_average_precision'] = float(data_total_precision) / float(
        total_data)
    # result['average_inferencing_time'] = float(
    #     data_total_inference_time) / float(total_data)
    result['total_image'] = total_data
    result['method'] = type_model
    return result

if __name__ == '__main__':
    args = getArgs()

    # 'mobilenet_ssd', 'opencv','mtcnn'
    list_data = get_data_DataIDFull()
    # print(list_data)
    type_model = 'mtcnn'
    result = calculate_metric(args, list_data, type_model)
    print('Average IOU  \t  mAP  \t  total_image  ')
    print( math.ceil(result['average_iou']*10000)/10000, "\t       ", result['mean_average_precision'], "\t",   math.ceil(result['total_image']*100)/100)
