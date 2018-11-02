from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
# import compare_list
# import align.detect_face
import cv2
import time
import os
import threading
import pickle
import queue
import align.detect_face
import facePre

color = (0, 255, 0)  # 绿色
cameraIsOpened = False
tracker_queue = queue.Queue()


class Tracker(object):
    'Tracker class definition'

    def __init__(self, frame, bound_box, emb):
        self.tracker = cv2.TrackerCSRT_create()
        self.emb = emb
        self.id = -1
        self.imgs = []
        self.time = time.ctime()
        self.noface_cnt = 0
        self.person_in = False
        self.path = []
        ok = self.tracker.init(frame, bound_box)
        if(ok):
            print("tracker initialized!")
        else:
            print("Initialize fail")

    def update(self, frame, tracker_list, classifier):
        print(self.id)
        h, l, d = frame.shape
        ok, bbox = self.tracker.update(frame)

        if(ok and self.noface_cnt <= 5):
            """
            # 判断追踪边框是否离开摄像头 如果离开, 判断是从哪里离开
            # 如果从摄像头下边框或者右边框左边框离开, 表示正确进入房间
            if(bbox[0] + bbox[2] >= l or bbox[1] + bbox[3] >= h):
                print("tracker effective")
                return True
            # 如果从上边框离开, 表示人并未进入房间, 这个追踪器无效
            elif(bbox[1] <= 0 and len(self.imgs) >= 4):
                print("tracker invalid")align
                return False
            """
            # 对每隔追踪结果画框并且显示追踪器标号, 并且把当前追踪的结果加入到追踪器照片列表中
            box = []
            for i in bbox:
                box.append(int(i))
            # 计算当前追踪器在y轴的位置
            point_y = int(box[1] + box[3] / 2)
            if(point_y > 600):
                self.person_in = True
            image = frame[box[1]-20:box[1]+box[3]+100, box[0]-20: box[0]+box[1]+100]
            self.imgs.append(image)
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceRects = classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(120, 120))
            if(len(faceRects) > 0):
                # cv2.putText(frame, str(self.id), (box[0], box[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                self.noface_cnt = 0
            else:
                self.noface_cnt += 1
            return bbox
        else:
            path = self.time
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            for i in range(len(self.imgs)):
                cv2.imwrite(path + "/" + str(i) + ".jpg", self.imgs[i])
            tracker_queue.put(self)
            tracker_list.trackers[self.id] = None
            print("Delet sucessfully!")

            # 显示路径
            # self.show_tracker_path()
            if(self.person_in):
                t = threading.Thread(target=facePre.predict, args=([self.imgs]))
                t.start()

    def show_tracker_path(self):
        canvas = np.zeros((720, 1280, 3))
        green = (0, 255, 0)
        for i in range(len(self.path) - 1):
            cv2.line(canvas, self.path[i], self.path[i + 1], green)
        cv2.imshow("Canvas", canvas)


class Tracker_list(object):
    'Tracker_list class defination'

    def __init__(self):
        self.trackers = []

    def check_face(self, emb, frame, bbox):
        for tracker in self.trackers:
            if(tracker is not None):
                dis = np.sum(np.square(np.subtract(emb, tracker.emb)))
                print(dis)
                if(dis < 0.8):
                    tracker.tracker.init(frame, bbox)
                    tracker.emb = emb
                    return False
        else:
            return True

    def add_tracker(self, tracker):
        for i in range(len(self.trackers)):
            if(self.trackers[i] is None):
                tracker.id = i
                self.trackers[i] = tracker
                return
        else:
            tracker.id = len(self.trackers)
            self.trackers.append(tracker)

    def rm_tracker(self, tracker):
        self.trackers[tracker.id] = None


def dect_face():
    window_name = 'FaceRecognization'
    cv2.namedWindow(window_name)
    # 获取摄像头
    cap = catch_ip_camera("192.168.1.101")
    if(not cap):
        return

    # 使用人脸识别分类器
    classifier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")

    # config = tf.ConfigProto(device_count={"CPU": 1}, inter_op_parallelism_threads=1, intra_op_parallelism_threads=4,
    #                        log_device_placement=True)

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # load the model
            facenet.load_model("models/20180402-114759/20180402-114759.pb")
            # facenet.load_model("models/20170512-110547/20170512-110547.pb")
            # 创建一个追踪器列表对象
            tracker_list = Tracker_list()

            # 记录当前为第几帧
            frame_cnt = 0

            # 设置张量
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # 如果视屏流正常, 则开始操作
            while(cap.isOpened()):

                ret, frame = cap.read()
                if(ret is True):
                    frame_cnt += 1

                    # 每5帧检测一次人脸
                    if(frame_cnt % 5 == 0):
                        # ts = time.time()
                        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faceRects = classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(130, 130))
                        if(len(faceRects) > 0):
                            threads = []
                            for faceRect in faceRects:
                                t = threading.Thread(target=face_dect, args=(frame, faceRect, sess, tracker_list,
                                                                             images_placeholder, phase_train_placeholder,
                                                                             embeddings))
                                threads.append(t)
                            for t in threads:
                                t.start()
                            for t in threads:
                                t.join()
                        # tp = time.time()
                        # print("dect time", tp - ts)

                    # 每10帧进行一次跟踪, 采用多线程的方式, 希望可以提高多个追踪器更新时的效率
                    if(frame_cnt % 2 == 0):
                        # ts = time.time()
                        threads = []
                        for tracker in tracker_list.trackers:
                            if(tracker is not None):
                                t = threading.Thread(target=tracker.update, args=(frame, tracker_list, classifier))
                                threads.append(t)
                        for t in threads:
                            t.start()
                        for t in threads:
                            t.join()
                        # tp = time.time()
                        # print("update time", tp - ts)

                    cv2.imshow(window_name, frame)
                    c = cv2.waitKey(1)
                    if(c & 0xFF == ord('q')):
                        break

                else:
                    print("The connection is break down!")
                    cap.release()
                    cap = catch_ip_camera("192.168.1.101")
                    ret, frame = cap.read()
                frame_cnt = frame_cnt % 10000
    cap.release()
    cv2.destroyAllWindows()


def catch_ip_camera(ip):
    # 设置获取ip地址
    ip = "rtsp://admin:admin123@" + ip + "//Streaming/Channels/1"
    # 获取摄像头
    cap = cv2.VideoCapture(ip)
    if(cap.isOpened()):
        print("Connected!")
        return cap
    else:
        print("Fail to connect!")
        return False


def face_dect(frame, faceRect, sess, tracker_list, images_placeholder, phase_train_placeholder, embeddings):
    x, y, w, h = faceRect
    bound_box = (x, y, w, h)
    image = frame[y - 22:y + h + 22, x - 22: x + w + 22]

    feed_dict = {images_placeholder: pre_process_img(image), phase_train_placeholder: False}
    embs = sess.run(embeddings, feed_dict=feed_dict)
    emb = embs[0]

    # cv2.rectangle(frame, (x - 22, y - 22), (x + w + 22, y + h + 22), color, 2)
    if(tracker_list.check_face(emb, frame, bound_box)):
        tracker = Tracker(frame, bound_box, emb)
        tracker.imgs.append(image)
        tracker_list.add_tracker(tracker)
        print(tracker_list.trackers)


def caculate_one_pic(image, sess):
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_load_and_align_datagraph().get_tensor_by_name("phase_train:0")

    feed_dict = {images_placeholder: pre_process_img(image), phase_train_placeholder: False}
    embs = sess.run(embeddings, feed_dict=feed_dict)
    emb = embs[0]
    return emb


def pre_process_img(Image):
    pics_ls = []
    processed = cv2.resize(Image, (160, 160))
    processed = processed[:, :, ::-1]
    prewhitened = facenet.prewhiten(processed)
    pics_ls.append(prewhitened)
    return pics_ls


def get_pics_in_facelist():
    path = 'facelist_pics/'
    ls = os.listdir(path)
    pic_id_ls = []
    for i in ls:
        Img = misc.imread(path + i)
        Img = facenet.prewhiten(Img)
        pic_id_ls.append(Img)

    return pic_id_ls


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


def predict(images):
    with tf.Graph().as_default():

        with tf.Session() as sess:
            # Load the model
            facenet.load_model("models/20180402-114759/20180402-114759.pb")

            # Get input and out put tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Read classifier file
            classifier_filename_exp = os.path.expanduser("fourPeopleClassifier.pkl")
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            images = load_and_align_data(images)
            print(len(images))
            # Run forward pass to caculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)
            predictions = model.predict_proba(emb)
            print(predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            # best_class_probabilities = predictions[np.arrange(len(best_class_indices)), best_class_indices]
            for i in best_class_indices:
                print("best_class_indices:", i)


def load_and_align_data(images, image_size=160, margin=44, gpu_memory_fraction=1.0):

    print("start load!")
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    img_list = []
    for img in images:
        img = img[:, :, ::-1]
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        for j in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[j, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


if __name__ == '__main__':
    t1 = threading.Thread(target=dect_face, args=())
    # t2 = threading.Thread(target=predict, args=())
    t1.start()
    # t2.start()
