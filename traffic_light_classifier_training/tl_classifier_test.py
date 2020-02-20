import cv2
import tensorflow as tf
import numpy as np
import glob
from keras.models import load_model

RED_IMAGE_PATH = '/home/workspace/teamwork/dataset/simulator_0220_3/has_light/red/'
YELLOW_IMAGE_PATH = '/home/workspace/teamwork/dataset/simulator_0220_3/has_light/yellow/'
GREEN_IMAGE_PATH = '/home/workspace/teamwork/dataset/simulator_0220_3/has_light/green/'
NONE_IMAGE_PATH = '/home/workspace/teamwork/dataset/simulator_0220_3/no_light/'
MODEL_FILE = '/home/workspace/teamwork/traffic_light_classifier_training/tl_classifier_simulator.h5'


class TLClassifierChecker:
    def __init__(self):
        #TODO load classifier
        self.graph = tf.get_default_graph()
        self.model = None
        self.test_data = []
        self.test_result = []

    def set_model(self, model_path):
        model = load_model(model_path)
        self.model = model

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
        #TODO implement light color prediction
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(image / 255., (60, 80))
        with self.graph.as_default():
            predictions = self.model.predict(resized_img.reshape((1, 80, 60, 3)))
            result = predictions[0].tolist().index(np.max(predictions[0]))
            return result


checker = TLClassifierChecker()
checker.set_model(MODEL_FILE)
checker.initialize_test_data()
checker.conduct_test()
checker.show_test_result()

