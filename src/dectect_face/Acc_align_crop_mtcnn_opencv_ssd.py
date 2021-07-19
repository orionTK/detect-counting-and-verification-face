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
import csv
import glob
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/DataIDFull', help='the dir your dataset of face which need to crop')
    parser.add_argument('--output_path', type=str, default='C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/DataIDFullCroped', help='the dir the cropped faces of your dataset where to save')
    # parser.add_argument('--face-num', '-face_num', type=int, default=1, help='the max faces to crop in each image')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id， when the id == -1, use cpu')
    parser.add_argument('--face_size', type=str, default='256', help='the size of the face to save, the size x%2==0, and width equal height')
    args = parser.parse_args()
    return args


def get_iou(boxA, boxB):
    """
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	boxA = np.array( [ xmin,ymin,xmax,ymax ] )
	boxB = np.array( [ xmin,ymin,xmax,ymax ] )

	Returns
	-------
	float
		in [0, 1]
	"""

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

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes area
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def extract_and_filter_data(splits):
    # Extract bounding box ground truth from dataset annotations, also obtain each image path
    # and maintain all information in one dictionary
    bb_gt_collection = dict()

    for split in splits:

        with open(os.path.join('DataTest', 'wider_face_split', 'wider_face_%s_bbx_gt.txt' % (split)), 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.split('\n')[0]
            if line.endswith('.jpg'):
                image_path = os.path.join('DataTest', 'WIDER_%s' % (split),
                                          'images', line)
                bb_gt_collection[image_path] = []
            line_components = line.split(' ')
            if len(line_components) > 1:

                # Discard annotation with invalid image information, see dataset/wider_face_split/readme.txt for details
                if int(line_components[7]) != 1:
                    x1 = int(line_components[0])
                    y1 = int(line_components[1])
                    w = int(line_components[2])
                    h = int(line_components[3])

                    # In order to make benchmarking more valid, we discard faces with width or height less than 15 pixel,
                    # we decide that face less than 15 pixel will not informative enough to be detected
                    if w > 15 and h > 15:
                        # print("test: ", image_path, line_components)
                        bb_gt_collection[image_path].append(
                            np.array([x1, y1, x1 + w, y1 + h]))

    return bb_gt_collection

# lay label da dan cho bo du lieu DataIDFull
def get_data_DataIDFull(path):
    i = 0
    bb_gt_collection = dict()
    list_file_none = []

    all_file_image = glob.glob('../Image/DataIDFull/*/*.jpg')
    if all_file_image != []:
        for file in all_file_image:
            file = file.replace('\\', '/')
            name_img = (file.split('/')[-1])[0:(len(file.split('/')[-1]) - 4)]
            print(os.path.exists(
                os.path.join('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image', 'label', '%s.txt' % (name_img))))
            if not os.path.exists(os.path.join('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image', 'label',
                                               '%s.txt' % (name_img))):
                list_file_none.append(file)
                continue
            with open(os.path.join('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image', 'label',
                                   '%s.txt' % (name_img)), 'r') as f:
                lines = f.readlines()

            for line in lines:
                i += 1
                line = line.split('\n')[0]

                image_path = file
                if cv2.imread(image_path) is None:
                    continue

                # print(cv2.imread(image_path))
                bb_gt_collection[image_path] = []
                line_components = line.split(' ')
                print('....................', line)
                if len(line_components) > 1:
                    # print(line_components[1])
                    img = cv2.imread(image_path)
                    img_h, img_w, d = img.shape
                    print('-----------------test', img_w, '==================', img_h)
                    x1 = float(line_components[1].replace(',', '.')) * img_w
                    y1 = float(line_components[2].replace(',', '.')) * img_h
                    w = float(line_components[3].replace(',', '.')) * img_w
                    h = float(line_components[4].replace(',', '.')) * img_h
                    # print(image_path)
                    # print('h', np.array([x1, y1, x1 + w, y1 + h]))
                    # In order to make benchmarking more valid, we discard faces with width or height less than 15 pixel,
                    # we decide that face less than 15 pixel will not informative enough to be detected
                    if w > 15 and h > 15:
                        # print("test: ", image_path, line_components)

                        bb_gt_collection[image_path].append(
                            np.array([x1 - w / 2, y1 - h / 2, x1 + w / 2, y1 + h / 2]))
                    # print('-----', bb_gt_collection)

                print('len = ', i)
    np.save('filenone.npy', list_file_none)
    print(len(bb_gt_collection))
    return bb_gt_collection

# cũ
def crop_align_face(args, method):
    input_dir = args.input_path
    output_dir = args.output_path
    # face_num = args.face_num
    total_face = 0
    if not os.path.exists(input_dir):
        print('the input path is not exists!')
        sys.exit()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    # code cũ
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
    #
    mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)

    count_no_find_face = 0
    count_crop_images = 0

    # code cũ

    for root, dirs, files in tqdm(os.walk(input_dir)):
        print(root)
        output_root = root.replace(input_dir, output_dir)
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        j = 0
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.split('.')[-1].lower() == 'jpg':

                face_img = cv2.imread(file_path)
                print(face_img)
                print(face_img)

                ret = mtcnn.detect_face(face_img)
                print(ret)
                if ret is None:
                    print('%s do not find face'%file_path)
                    count_no_find_face += 1
                    continue
                bbox, points = ret
                if bbox is None:
                    print('%s do not find face'%file_path)
                    count_no_find_face += 1
                    continue
                print(bbox)
                name = '%d.jpg' % (j)
                j += 1
                for i in range(bbox.shape[0]):
                    bbox_ = bbox[i, 0:4]
                    print("meo: ", bbox)
                    # crop
                    points_ = points[i, :].reshape((2, 5)).T
                    face = mtcnn.preprocess(face_img, bbox_, points_, image_size=args.face_size)

                    face_name = '%s_%d.jpg'%(name.split('.')[0], i)
                    # print(input_dir)
                    # face_name = '%s_%d.jpg' % (output_root.split('/')[-1],i)

                    # print(face_name)
                    file_path_save = os.path.join(output_root, face_name)
                    cv2.imwrite(file_path_save, face)
                    cv2.imshow('face', face)
                    cv2.waitKey(0)
                    total_face += 1
                count_crop_images += 1
            print('done!')
    print('%d images crop successful!' % count_crop_images)
    print('%d images do not crop successful!' % count_no_find_face)

def calculate_metric(args, bb_gt_collection, type_model):
    output_dir = args.output_path
    total_data = len(bb_gt_collection.keys())
    print('===============================================', bb_gt_collection, "+++++++++++++++++++++++++++++++++++++++")
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
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
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
        print('out_path', out_path)
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
        print("image_data ==============", image_data)
        # đã gán
        face_bbs_gt = np.array(bb_gt_collection[key])
        total_gt_face = len(face_bbs_gt)

        start_time = time.time()
        ret = detect.detect_face(image_data)
        inf_time = time.time() - start_time
        data_total_inference_time += inf_time
        print("----------------------------------------------------------")
        print('ret', ret)
        if ret is None:
            print('Do not find face')
            continue
        if type_model == 'mtcnn':
            face_pred, points = ret
        else:
            face_pred = ret

        ### Calc average IOU, Precision, and Average inferencing time ####
        total_iou = 0
        tp = 0
        pred_dict = dict()
        for gt in face_bbs_gt:
            max_iou_per_gt = 0
            # print("face: ", gt[0], gt[1], gt[2], gt[3])
            # cv2.rectangle(image_data, (gt[0], gt[1]), (gt[2], gt[3]),
            #               (255, 0, 0), 2)
            for i, pred in enumerate(face_pred):
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++", key)
                print("predict: ", pred[0], pred[1], pred[2], pred[3])
                print("gt: ", gt)
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                # cv2.rectangle(image_data, (int(pred[0]), int(pred[1])),(int(pred[2]), int(pred[3])), (0, 0, 255), 2)
                name_img = key.split('/')[-1]

                file_path = '%s_%d.jpg' % (name_img.split('.')[0], i)
                print('save: ', file_path)
                file_path_save = os.path.join(filename, file_path)

                print('file: ', file_path_save.replace('\\', '/'))
                # cv2.imwrite(file_path_save.replace('\\', '/'), face)
                # cv2.imwrite(file_path_save.replace('\\', '/'), face)
                iou = get_iou(gt, pred)
                # cv2.imshow("meo", image_data)
                # cv2.imwrite("meo1.jpg", image_data)
                # cv2.waitKey(0)
                print('iou', iou)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt

        if total_gt_face != 0:
            if len(pred_dict.keys()) > 0:
                for i in pred_dict:
                    print('pred', pred_dict[i])
                    if pred_dict[i] >= 0.5:
                        tp += 1
                precision = float(tp) / float(total_gt_face)

            else:
                precision = 0

            image_average_iou = total_iou / total_gt_face
            image_average_precision = precision

            data_total_iou += image_average_iou
            data_total_precision += image_average_precision

        ## crop
        if type_model == "mtcnn":
            for i in range(face_pred.shape[0]):
                bbox_ = face_pred[i, 0:4]
                # print("meo: ", face_pred)
                # cv2.rectangle(image_data, (int(pred[0]), int(pred[1])),(int(pred[2]), int(pred[3])), (225, 0, 0), 2)

                # crop
                points_ = points[i, :].reshape((2, 5)).T
                face = mtcnn.preprocess(image_data, bbox_, points_, image_size=args.face_size)
                name_img = key.split('/')[-1]

                file_path = '%s_%d.jpg' % (name_img.split('.')[0], i)
                print('save: ', file_path)
                file_path_save = os.path.join(filename, file_path)

                print('file: ', file_path_save.replace('\\', '/'))
                cv2.imwrite(file_path_save.replace('\\', '/'), face)
        print('done!')
    result = dict()
    result['average_iou'] = float(data_total_iou) / float(total_data)
    result['mean_average_precision'] = float(data_total_precision) / float(
        total_data)
    result['average_inferencing_time'] = float(
        data_total_inference_time) / float(total_data)
    result['totall_image'] = total_data
    result['method'] = type_model
    return result

if __name__ == '__main__':
    args = getArgs()
    pre = OpenCVHaarFaceDetector(scaleFactor=1.3,
                                 minNeighbors=5,
                                 model_path='models/haarcascade_frontalface_default.xml')
    # crop_align_face(args)
    # result = calculate_metric(args, extract_and_filter_data(['train', 'val']), 'mtcnn')
    # 'mobilenet_ssd', 'opencv','mtcnn'
    method = ['mobilenet_ssd']
    with open("test.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=('model', 'data', 'Average_IOU', "mAP", "total_data", 'average_time'))
        writer.writeheader()
        list_data = get_data_DataIDFull('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/DataIDFull/')
        # list_data = extract_and_filter_data(['train', 'val'])
        for i in method:
            result = calculate_metric(args, list_data, i)
        # result = calculate_metric(args, get_data_DataIDFull('C:/Users/SV_Guest/Desktop/Kieu/DoAn/src/Image/DataIDFull/'), i)
            writer.writerow({"model": i, "data": "DataIDFull", 'Average_IOU': str(round(result['average_iou'] * 100, 2)),"mAP": str(round(result['mean_average_precision'] * 100, 2)),
            "total_data": str(result['totall_image']), "average_time": str(result['average_inferencing_time'])} )


    # hiệu suất mtcnn ban đầu cao hơn
    print('=== method  \t  Data  \t  Average IOU  \t  mAP  \t  total_data  \t  Avg_Time ')

    with open("test.csv", "r") as f:
        results = csv.DictReader(f)
        for result in results:
            # print('Average IOU = %s%s' % (str(round(result['average_iou'] * 100, 2)), '%'))
            # print('mAP = %s%s' % (str(round(result['mean_average_precision'] * 100, 2)), "%"))
            # print('Average inference time = %s' % (str(result['average_inferencing_time'])))
            # print('Total face  = %s' % (str(result['totall_image'])))

            # print("done!")

            print(' %s  \t  %s  \t  %s%s \t  %s%s  \t  %s  \t  %s '
                  % (result['model'], 'Wider Face', result['Average_IOU'], '%'
                     , result['mAP'], "%", result['total_data'],
                     result['average_time']))
            print('==========================================================================================')
