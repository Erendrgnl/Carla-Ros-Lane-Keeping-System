import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from threading import Thread



def callback(image):
    byte_image = image.data
    np_image = np.frombuffer(byte_image,dtype=np.uint8)
    bgra_image = np_image.reshape((image.height,image.width,4))
    bgr_image = cv2.cvtColor(bgra_image,cv2.COLOR_BGRA2BGR)
    

    cv2.imshow("Camera Front",bgr_image)
    cv2.waitKey(10)

if __name__ == "__main__":
    rospy.init_node('camera', anonymous=True)

    rospy.Subscriber("/lka/detected_image", Image, callback)

    Thread(target=rospy.spin()).start()
