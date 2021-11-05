from operator import le
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from threading import Thread

import os
from laneNet.tools import laneNet_class

net = None
#center = (256,106)
def bev(image):
    IMAGE_H = 256
    IMAGE_W = 512
    V_POINT = 140
    src = np.float32([[0, IMAGE_H], [512, IMAGE_H], [IMAGE_W, V_POINT],[0, V_POINT]])
    dst = np.float32([[237, IMAGE_H], [275, IMAGE_H], [IMAGE_W, 0],[0, 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    #cv2.polylines(image,[src.astype(np.int32)],True,(0,0,255),thickness=2)
    M = cv2.getPerspectiveTransform(src,dst)
    out = cv2.warpPerspective(image,M,(IMAGE_W, IMAGE_H),flags=cv2.INTER_LINEAR)
    out = out[150:IMAGE_H,0:IMAGE_W,:]
    return out

def postProcess(image):
    src_img = image.copy()
    
    red_mask = (src_img[:,:,2]>200).astype(np.uint8)
    src_img = cv2.bitwise_and(src_img,src_img,mask=1-red_mask)
    
    #Right Lanes
    green_mask = (src_img[:,:,1]>200).astype(np.uint8)
    green_area = cv2.bitwise_and(src_img,src_img,mask=green_mask)

    #Left Lanes
    blue_mask = (src_img[:,:,0]>200).astype(np.uint8)
    blue_area = cv2.bitwise_and(src_img,src_img,mask=blue_mask)

    all_lanes = cv2.addWeighted(green_mask,1,blue_mask,1,0)
    return all_lanes

def window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
 
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    #Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 5
    # Set minimum number of pixels found to recenter window
    minpix = 15

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_y_center = win_y_low + int((win_y_high-win_y_low)/2)
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Generate black image and colour lane lines
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
        
    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (255,255,0), thickness=5)
    cv2.polylines(out_img, [left], False, (255,255,0), thickness=5)
    
    return left_lane_inds, right_lane_inds, out_img

def callback(image):
    global net

    byte_image = image.data
    np_image = np.frombuffer(byte_image,dtype=np.uint8)
    bgra_image = np_image.reshape((image.height,image.width,4))
    bgr_image = cv2.cvtColor(bgra_image,cv2.COLOR_BGRA2BGR)
    
    lane_mask = net.predict(cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB))
    bgr_image = cv2.resize(bgr_image,(512,256))
    lane_mask = cv2.cvtColor(lane_mask,cv2.COLOR_BGRA2BGR)
    bev_img = bev(lane_mask)
    r = postProcess(bev_img)
    _,_,out = window_search(r)
    #window_search(r)

    cv2.imshow("Camera Front",bgr_image)
    cv2.imshow("Camera Front BEV",bev_img)
    cv2.imshow("Right Lane",r)
    cv2.imshow("Out",out)
    cv2.imshow("Camera Front Lane Mask",lane_mask.astype(np.uint8))
    cv2.waitKey(10)

if __name__ == "__main__":
    weights_path = r"/home/eren/Codes/Python/LaneNet/lanenet-lane-detection/weights/tusimple_lanenet.ckpt"
    net = laneNet_class.LaneNet(weights_path)
    rospy.init_node('eren_camera', anonymous=True)

    rospy.Subscriber("/carla/ego_vehicle/rgb_front/image", Image, callback)

    Thread(target=rospy.spin()).start()
