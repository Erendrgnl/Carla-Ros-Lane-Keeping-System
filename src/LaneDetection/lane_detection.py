import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from threading import Thread

from LaneDetection.models import laneNet_class

net = None

def window_search(righ_lane, left_lane):
    out = np.zeros(righ_lane.shape)
    out = cv2.merge((out,out,out))

    mid_point = np.int(righ_lane.shape[1]/2)
    #out[:,mid_point]=(0,255,0)

    nwindows = 9
    h = righ_lane.shape[0]
    vp = int(h/2)
    window_height = np.int(vp/nwindows)

    r_lane = righ_lane[vp:,:].copy()
    r_lane = cv2.erode(r_lane,np.ones((3,3)))

    l_lane = left_lane[vp:,:]
    l_lane = cv2.erode(l_lane,np.ones((3,3)))
    
    for window in range(nwindows):
        win_y_low = vp - (window+1)*window_height
        win_y_high = vp - window*window_height
        win_y_center = win_y_low + int((win_y_high-win_y_low)/2)

        r_row = r_lane[win_y_low:win_y_high,:]
        l_row = l_lane[win_y_low:win_y_high,:]

        histogram = np.sum(r_row, axis=0)
        r_point = np.argmax(histogram)
        #if(r_point != 0):
        #    out = cv2.circle(out,(r_point,vp+win_y_center),5,(0,0,255),-1)

        histogram = np.sum(l_row, axis=0)
        l_point = np.argmax(histogram)
        #if(l_point != 0):
        #    out = cv2.circle(out,(l_point,vp+win_y_center),5,(0,0,255),-1)

        if(l_point != 0) and (r_point != 0):
            rd = r_point-mid_point
            ld = mid_point-l_point
            if(abs(rd-ld)<100):
                center = l_point + int((r_point-l_point)/2)
                out = cv2.circle(out,(center,vp+win_y_center),2,(0,0,255),-1)
            #print("R Dist: ",rd)
            #print("L Dist: ", ld)
            #print("D: ",abs(rd-ld))
    return out

def get_lines(image):
    src_img = image.copy()
    
    red_mask = (src_img[:,:,2]>200).astype(np.uint8)
    src_img = cv2.bitwise_and(src_img,src_img,mask=1-red_mask)
    
    #Right Lanes
    green_mask = (src_img[:,:,1]>200).astype(np.uint8)
    green_area = cv2.bitwise_and(src_img,src_img,mask=green_mask)

    #Left Lanes
    blue_mask = (src_img[:,:,0]>200).astype(np.uint8)
    blue_area = cv2.bitwise_and(src_img,src_img,mask=blue_mask)

    lanes_rgb = cv2.addWeighted(green_area,1,blue_area,1,0)
    return lanes_rgb,(green_mask*255),(blue_mask*255)


def callback(image):
    global net

    byte_image = image.data
    np_image = np.frombuffer(byte_image,dtype=np.uint8)
    bgra_image = np_image.reshape((image.height,image.width,4))
    bgr_image = cv2.cvtColor(bgra_image,cv2.COLOR_BGRA2BGR)
    
    prediction = net.predict(cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB))
    bgr_image = cv2.resize(bgr_image,(prediction.shape[1],prediction.shape[0]))
    
    prediction = cv2.cvtColor(prediction,cv2.COLOR_BGRA2BGR)
    lanes_rgb,right_lane,left_lane = get_lines(prediction)
    rgb_result = cv2.addWeighted(bgr_image,1,lanes_rgb,0.6,0)

    asd = window_search(right_lane,left_lane).astype(np.uint8)
    asd = cv2.addWeighted(rgb_result,1,asd,0.6,0)

    cv2.imshow("Camera Front",rgb_result)
    cv2.imshow("Lane Detection",asd)
    cv2.waitKey(10)

if __name__ == "__main__":
    weights_path = r"/home/eren/Codes/Python/LaneNet/lanenet-lane-detection/weights/tusimple_lanenet.ckpt"
    net = laneNet_class.LaneNet(weights_path)
    rospy.init_node('eren_camera', anonymous=True)

    rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, callback)

    Thread(target=rospy.spin()).start()
