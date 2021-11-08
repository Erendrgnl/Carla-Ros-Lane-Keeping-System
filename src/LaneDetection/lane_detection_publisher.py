import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2

class LaneDetection(object):
    def __init__(self,model):
        self.model = model
        rospy.init_node('camera', anonymous=True)
        rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, self.callback)
        self.pub = rospy.Publisher("/lka/detected_image",Image,queue_size=10)
        rospy.spin()

    def callback(self,raw_image):
        byte_image = raw_image.data
        np_image = np.frombuffer(byte_image,dtype=np.uint8)
        bgra_image = np_image.reshape((raw_image.height,raw_image.width,4))
        rgb_image = cv2.cvtColor(bgra_image,cv2.COLOR_BGRA2RGB)
        
        publish_image = Image()
        publish_image.header = raw_image.header
        publish_image.is_bigendian = raw_image.is_bigendian
        publish_image.encoding = raw_image.encoding

        prediction,lane_center = self.model.predict(rgb_image)
        publish_image.height = prediction.shape[0]
        publish_image.width  = prediction.shape[1]

        prediction = cv2.cvtColor(prediction,cv2.COLOR_RGB2BGRA).astype(np.uint8)
        byte_data = prediction.tobytes()
        publish_image.data = byte_data

        self.pub.publish(publish_image)


if __name__ == "__main__":
    from LaneDetection.models import laneNet_class
    
    model = laneNet_class.LaneNet()
    ros_node = LaneDetection(model)