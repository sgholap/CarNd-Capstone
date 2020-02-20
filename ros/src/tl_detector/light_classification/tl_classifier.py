from styx_msgs.msg import TrafficLight
import cv2
import tensorflow as tf
import rospy
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.graph = tf.get_default_graph()
        self.model = None

    def set_model(self, model):
        self.model = model

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img / 255., (60, 80))
        with self.graph.as_default():
            predictions = self.model.predict(resized_img.reshape((1, 80, 60, 3)))
            result = predictions[0].tolist().index(np.max(predictions[0]))
            traffic_light = TrafficLight()
            if result == 3: 
                # none_light
                traffic_light.state = 4
            else:
                traffic_light.state = result
            rospy.loginfo("model output: %s", traffic_light.state)
        return traffic_light.state
