from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
import os
import pickle
import align.detect_face


def predict(images):
    print("Start predict!")
    with tf.Graph().as_default():

        with tf.Session() as sess:
            # Load the model
            facenet.load_model("models/20180402-114759/20180402-114759.pb")

            # Get input and out put tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Read classifier file
            classifier_filename_exp = os.path.expanduser("twoPeopleClassifier.pkl")
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
