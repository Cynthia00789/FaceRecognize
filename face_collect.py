from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
import compare_list
import cv2
import time
import os


def collect():
    window_name = 'FaceCollection'
    cv2.namedWindow(window_name)
    # 获取摄像头
    cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.101//Streaming/Channels/1")
    if(cap.isOpened()):
        print("Connected!")
    else:
        print("Fail to connect!")
    # 使用人脸识别分类器
    classifier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")

    color = (0, 255, 0)  # 边框颜色
    t1 = t2 = time.time()
    pics_ls = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret is True):
            cnt = 0
            t2 = time.time()
            if((t2 - t1) * 1000 >= 100):
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faceRects = classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if(len(faceRects)):
                    face_in = True
                    for faceRect in faceRects:
                        x, y, w, h = faceRect
                        cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, 2)
                        cpimg = frame[y - 10:y + h + 10, x - 10:x + w + 10]
                        pics_ls.append(cpimg)
                t1 = t2
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(1)
            if(c & 0xFF == ord('q')):
                break
        else:
            print("The connection is break down!")
            cap.release()
            cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.101//Streaming/Channels/1")
            ret, frame = cap.read()
            face_cnt = 0
    cap.release()
    cv2.destroyAllWindows()

    return pics_ls


def pre_process_img(Images):
    """
    # 定义Tensorflow图
    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(pre_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    """

    pics_ls = []

    for Image in Images:
        """
        Image = i[:, :, ::-1]
        img_size = np.asarray(Image)[0:2]
        t1 = time.time()
        # faceRects, _ = align.detect_face.detect_face(Image, minsize, pnet, rnet, onet, threshold, factor)
        t2 = time.time()
        print(t2 - t1)
        if(len(faceRects) > 0):
            for i in range(len(faceRects)):
                det = np.squeeze(faceRects[i][0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-22, 0)
                bb[1] = np.maximum(det[1]-22, 0)
                bb[2] = np.minimum(det[2]+22, 1280)
                bb[3] = np.minimum(det[3]+22, 720)
                cropped = Image[bb[1]:bb[3], bb[0]:bb[2], :]

                aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                pics_ls.pics_lsappend(prewhitened)
        t3 = time.time()
        print(t3 - t2)
        """
        aligned = cv2.resize(Image, (160, 160), interpolation=cv2.INTER_CUBIC)
        aligned = aligned[:, :, ::-1]
        prewhitened = facenet.prewhiten(aligned)
        pics_ls.append(prewhitened)

    return pics_ls


def get_compare_result(emb, id_cnt, pics_cnt, face_cnt):
    cau_ls = [0]
    cau_ls *= id_cnt
    strangers_ls = []
    rec_ls = []

    # 对摄像头截取到的每一张图片, 都与facelist中的照片进行对比, 找到比阀值1.0小并且是最小的那一个值的图片标号,
    # 给对应标号的cau_ls数组+1, 用来后续判断
    # 如果所有照片比对后都不小于1.0, 那么就判断为陌生人, 加入strangers_ls列表中
    for i in range(id_cnt, pics_cnt):
        min_res = 10.0
        min_idx = 0
        in_facelist = False
        for j in range(0, id_cnt):
            res = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
            print("%.4f  " % res, end=' ')
            if(res < min_res and res < 1):
                in_facelist = True
                min_res = res
                min_idx = j
        if(in_facelist):
            cau_ls[min_idx] += 1
        else:
            strangers_ls.append(i)

        print(cau_ls)

    # 对于cau_ls中的每一个数, 如果大于6, 也就是在进门是捕获了6张照片且都判断为对应标号的人, 那么认为本次识别有效.
    # 如果被判断为这个人的次数小于等于6, 那么认为这是一次噪声判断.
    # idx = 0
    for j in range(len(cau_ls)):
        """maxn = 0
        if(cau_ls[j] > maxn):
            maxn = cau_ls[j]
            idx = j"""
        if(cau_ls[j] > 7):
            rec_ls.append(j)

    for i in range(len(rec_ls)):
        if(rec_ls[i] == 0):
            print("fyw")
        elif (rec_ls[i] == 1):
            print("zyf")

    for i in range(len(strangers_ls)):
        print("pic%d is a stranger" % strangers_ls[i])

    return rec_ls, strangers_ls


if __name__ == '__main__':
    print("请正对摄像头")
    col_ls = collect()
    col_ls = pre_process_img(col_ls)
    np_col_ls = np.stack(col_ls)
    # img = facenet.prewhiten(np_col_ls)
    # img = img[:, :, ::-1]
    cv2.imshow('img', img)
    cv2.waitKey(2000)
    print(np_col_ls[1])
