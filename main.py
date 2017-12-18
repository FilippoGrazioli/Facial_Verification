"""
Reel-time API that captures RGB frames of the driver and detects if their identity matches with one
known identity (see the known_identities folder).

A GUI shows the detected identity and the prediction accuracy in terms of a mean distance between the
cropped face's embedding and the most likely identity's embeddings.

Filippo Grazioli
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import facenet
import align.detect_face
from scipy import misc
import cv2
import tensorflow as tf
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import time
from scipy.spatial import distance
from bbox_helper import Rectangle


model_dir = 'model_checkpoints/20170512-110547'
identities_dir = 'known_identities'
unknown_id_distance_threshold = 1.1  # Threshold for determining whether a face is unknown
k_num = 4  # Number of neighbours for K-Nearest Neighbours Classifier
distance_buffer_length = 4


class KnownDriver:

    def __init__(self, known_driver):

        self.embeddings = np.load(os.path.join(identities_dir, known_driver, known_driver+'_embeddings.npy'))
        self.name = known_driver
        self.pics_number = len([name for name in os.listdir('.') if os.path.isfile(name)])-1

    def get_name(self):
        return self.name

    def get_embeddings(self):
        return self.embeddings


def get_known_drivers():
    known_drivers = [x[0] for x in os.walk(identities_dir)]
    known_drivers = [KnownDriver(known_driver.replace(identities_dir+'/', '')) for known_driver in known_drivers[1:]]
    return known_drivers


def get_embeddings_given_name(known_drivers, name):
    embeddings = None
    for known_driver in known_drivers:
        if str(known_driver.get_name()) == str(name[0]):
            embeddings = known_driver.get_embeddings()
    return embeddings


def face_detection(frame, image_size, margin, pnet, rnet, onet):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img = frame
    img_size = np.array(frame).shape[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if np.array(bounding_boxes).size != 0:  # if a face is found
        found = True
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
    else:  # if a face is not found
        found = False
        prewhitened = np.zeros(shape=(160, 44, 3))
        bounding_boxes = [[0, 0, 0, 0, 0]]

    return np.array([prewhitened]), np.squeeze(np.around(bounding_boxes)).astype(int), found


def get_known_drivers_labels(known_drivers):
    y = []
    for known_driver in known_drivers:
        y.append(known_driver.get_name())
        y.append(known_driver.get_name())
        y.append(known_driver.get_name())
        y.append(known_driver.get_name())
    return y


def no_tf_debug():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def distance_face_id(new_embedding, known_embeddings):
    dist = 0
    for i in range(known_embeddings.shape[0]):
        known_embedding = known_embeddings[i, :]
        dist += distance.euclidean(known_embedding, new_embedding)
    return dist/4


def add_distance(distance_buffer, i, id_distance):
    distance_buffer[i] = id_distance
    if i < 3:
        i += 1
    else:
        i = 0
    return distance_buffer, i


def KNNCLassification(embed, known_drivers, n_neighbours=4):
    x = np.array([np.array(known_driver.get_embeddings()) for known_driver in known_drivers]).reshape((20, 128))
    y = get_known_drivers_labels(known_drivers)
    neigh = KNeighborsClassifier(n_neighbors=n_neighbours)
    neigh.fit(x, y)
    id_pred = neigh.predict(embed)
    embeddings = np.array(get_embeddings_given_name(known_drivers, id_pred))
    knn_accuracy = neigh.predict_proba(embed)
    id_distance = distance_face_id(embed, embeddings)
    return id_pred, knn_accuracy, id_distance


def main(model_dir):

    # Get the list of known drivers
    known_drivers = get_known_drivers()

    no_tf_debug()
    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Load the face detector model
            print('Loading the Face detection model...')
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            # Load the Inception-ResNet v1 model and get input & output tensors
            print('Loading the Inception-ResNet v1 model...')
            facenet.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            cap = cv2.VideoCapture(0)
            i = 0
            distance_buffer = np.zeros(shape=(distance_buffer_length,))
            while True:
                ret, frame = cap.read()

                # Crop the face from the frame
                start_time = time.time()
                aligned_frame, bb, found = face_detection(frame, 160, 44, pnet, rnet, onet)
                end_time = time.time()
                tpf_face = round(end_time - start_time, 5)
                if not found:
                    print('*' * 50)
                    print('TPF face detection : {} ||| Face NOT detected!'.format(str(tpf_face)))
                    print('*' * 50)

                if found:
                    # Forward pass of the face through the FaceNet to compute the embedding
                    start_time = time.time()
                    feed_dict = {images_placeholder: aligned_frame, phase_train_placeholder: False}
                    embed = sess.run(embeddings, feed_dict=feed_dict)
                    end_time = time.time()
                    tpf_facenet = round(end_time - start_time, 5)
                    print('*' * 50)
                    print('TPF FaceNet embedding : {0} ||| TPF face detection : {1} ||| Face detected!'
                          .format(str(tpf_facenet), str(tpf_face)))

                    # K-Nearest Neighbour Classification for classifying the driver identity w.r.t the known identities
                    id_driver, _4nn_accuracy, id_distance = KNNCLassification(embed, known_drivers, n_neighbours=k_num)
                    distance_buffer, i = add_distance(distance_buffer, i, id_distance)
                    if all(dis < unknown_id_distance_threshold for dis in distance_buffer):
                        print('Detected ID : {0} ||| ID 4-NN Accuracy : {1} % ||| Mean distance from detected ID : {2}'.
                              format(str(id_driver[0]), np.amax(_4nn_accuracy)*100, str(id_distance)))
                        print('*' * 50)
                    else:
                        print('I MIGHT NOT KNOW YOU!')
                        id_driver = 'Unknown'
                    
                    # Add bounding box of the face
                    if bb.ndim > 1:  # this is here because sometimes there are two bounding boxes on top of each others
                        bb = bb[0, :]  #TODO: undertand why this happens and get rid of this line
                    bbox = Rectangle(np.amax(bb[0]), np.amax(bb[1]), np.amax(bb[2]), np.amax(bb[3]),
                                     label=str(id_driver))
                    bbox.draw(frame, draw_label=True, color=255, thickness=3)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

        return 0


if __name__ == '__main__':
    main(model_dir)
