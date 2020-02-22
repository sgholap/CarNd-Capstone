from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime
import cv2
import glob

RED_IMAGE_PATH = '/home/workspace/self_dataset/simulator_0220_3/has_light/red/'
YELLOW_IMAGE_PATH = '/home/workspace/self_dataset/simulator_0220_3/has_light/yellow/'
GREEN_IMAGE_PATH = '/home/workspace/self_dataset/simulator_0220_3/has_light/green/'
NONE_IMAGE_PATH = '/home/workspace/self_dataset/simulator_0220_3/no_light/'


class ClassifierTest:
    def __init__(self, is_sim):
        self.test_data = []
        self.test_result = []

        if is_sim:
            PATH_TO_GRAPH = 'model/ssd_sim/frozen_inference_graph.pb'
        else:
            PATH_TO_GRAPH = 'model/ssd_udacity/frozen_inference_graph.pb'

        self.graph = tf.Graph()
        self.threshold = .5

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name(
                'num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def initialize_test_data(self):
        for name in glob.glob(RED_IMAGE_PATH + '*.jpg'):
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.test_data.append((img, 0, name))
        for name in glob.glob(YELLOW_IMAGE_PATH + '*.jpg'):
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.test_data.append((img, 1, name))
        for name in glob.glob(GREEN_IMAGE_PATH + '*.jpg'):
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.test_data.append((img, 2, name))
        for name in glob.glob(NONE_IMAGE_PATH + '*.jpg'):
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.test_data.append((img, 3, name))

    def conduct_test(self):
        for test_case in self.test_data:
            test_result = self.get_classification(test_case[0])
            self.test_result.append(test_result)

    def show_test_result(self):
        total_count = len(self.test_result)
        correct_count = 0
        fail_case = []
        for i in range(len(self.test_result)):
            if self.test_data[i][1] == self.test_result[i]:
                correct_count += 1
            else:
                fail_case.append((self.test_data[i][2], self.test_result[i]))
        print('total count: {}, correct count: {}'.format(total_count, correct_count))
        print('the following images failed:')
        for i in range(len(fail_case)):
            print('file: {}, result: {}'.format(fail_case[i][0], fail_case[i][1]))

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            start = datetime.datetime.now()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})
            end = datetime.datetime.now()
            c = end - start
            print(c.total_seconds())

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # print('SCORES: ', scores[0])
        # print('CLASSES: ', classes[0])

        if scores[0] > self.threshold:
            if classes[0] == 1:
                # print('GREEN')
                return 2
            elif classes[0] == 2:
                # print('RED')
                return 0
            elif classes[0] == 3:
                # print('YELLOW')
                return 1

        return 3


checker = ClassifierTest(True)
checker.initialize_test_data()
checker.conduct_test()
checker.show_test_result()


