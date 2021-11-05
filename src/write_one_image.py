import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from threading import Thread



def callback(image):
    global net

    byte_image = image.data
    np_image = np.frombuffer(byte_image,dtype=np.uint8)
    bgra_image = np_image.reshape((image.height,image.width,4))
    bgr_image = cv2.cvtColor(bgra_image,cv2.COLOR_BGRA2BGR)
    
    if type(bgr_image) != type(None):
        cv2.imwrite("carla_sample.png",bgr_image)    


if __name__ == "__main__":
    rospy.init_node('eren_camera', anonymous=True)

    rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, callback)

    Thread(target=rospy.spin()).start()
