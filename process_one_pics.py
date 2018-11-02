from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import compare_list
import align.detect_face
import cv2
import time
import sys
import argparse


def pre_process_Img(Image_path):
    # 定义Tensorflow图
    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(pre_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    pic_name_ls = get_pics_in_dir(Image_path)

    for pic_name in pic_name_ls:
        Image = misc.imread(Image_path + pic_name)
        img_size = np.asarray(Image)[0:2]
        faceRects, _ = align.detect_face.detect_face(Image, minsize, pnet, rnet, onet, threshold, factor)
        if(len(faceRects) > 0):
            det = np.squeeze(faceRects[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-22, 0)
            bb[1] = np.maximum(det[1]-22, 0)
            bb[2] = np.minimum(det[2]+22, 1280)
            bb[3] = np.minimum(det[3]+22, 720)
            cropped = Image[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
            misc.imsave(Image_path + pic_name, aligned)


def get_pics_in_dir(path):
    ls = os.listdir(path)
    pic_name_ls = []
    for i in ls:
        pic_name_ls.append(i)
    return pic_name_ls

if __name__ == '__main__':
    path = sys.argv[1:]
    pre_process_Img(path[0])
