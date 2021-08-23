import tensorflow as tf
import util
from argparse import ArgumentParser
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import mxnet as mx

import pylab as pl
import time
import os
import sys
from scipy.special import expit
import glob
from align_mtcnn.mtcnn_detector import MtcnnDetector

MAX_INPUT_DIM = 5000.0


def evaluate(weight_file_path, output_dir=None, data_dir=None, img=None, list_imgs=None,
             prob_thresh=0.5, nms_thresh=0.1, lw=3, display=False,
             draw=True, save=True, print_=0):

    if type(img) != np.ndarray:
        one_pic = False
    else:
        one_pic = True

    if not output_dir:
        save = False
        draw = False

    # list of bounding boxes for the pictures
    final_bboxes = []

    # placeholder of input images. Currently batch size of one is supported.
    x = tf.placeholder(tf.float32, [1, None, None, 3])  # n, h, w, c

    # Create the tiny face model which weights are loaded from a pretrained model.
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')
    #
    gpu = -1
    if gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(gpu)
    model = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)
    # score_final = model.detect_face(x)

    #
    #
    # # Reference boxes of template for 05x, 1x, and 2x scale
    # clusters = model.get_data_by_key("clusters")
    # clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    # clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    # normal_idx = np.where(clusters[:, 4] == 1)

    # Find image files in data_dir.
    filenames = []
    # if we provide only one picture, no need to list files in dir
    if one_pic:
        filenames = [img]
    elif type(list_imgs) == list:
        filenames = list_imgs
    else:
        for ext in ('*.png', '*.gif', '*.jpg', '*.jpeg'):
            filenames.extend(glob.glob(os.path.join(data_dir, ext)))

    # main
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for filename in filenames:
            # if we provide only one picture, no need to list files in dir
            if not one_pic and type(list_imgs) != list:
                fname = filename.split(os.sep)[-1]
                raw_img = cv2.imread(filename)
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            else:
                fname = 'current_picture'
                raw_img = filename
            raw_img_f = raw_img.astype(np.float32)



    if len(final_bboxes) == 1:
        final_bboxes = final_bboxes[0]
    return final_bboxes