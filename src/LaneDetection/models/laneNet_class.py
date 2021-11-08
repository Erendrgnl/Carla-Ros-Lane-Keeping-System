import cv2
import numpy as np
import tensorflow as tf
import os
import sys

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_PATH,"LaneNet"))

import lanenet
#from lanenet_model import lanenet_postprocess
import parse_config_utils


class LaneNet(object):
    def __init__(self):
        self.cfg = parse_config_utils.lanenet_cfg
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.net = lanenet.LaneNet(phase='test', cfg=self.cfg)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')


        self.weights_path = os.path.join(ROOT_PATH,"LaneNet","weights/tusimple_lanenet.ckpt")

        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = self.cfg.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = self.cfg.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)

        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                self.cfg.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        self.saver = tf.train.Saver(variables_to_restore)
        self.saver.restore(sess=self.sess, save_path=self.weights_path)
        
        print("LaneNet Model Initilaized")

    @staticmethod
    def preProcessing(image):
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        return image


    def predict(self,image):
        src_image = self.preProcessing(image)
        
        with self.sess.as_default():
            self.binary_seg_image, self.instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [src_image]}
            )
            rgb = self.instance_seg_image[0].astype(np.uint8)
            bw = self.binary_seg_image[0].astype(np.uint8)
            res = cv2.bitwise_and(rgb,rgb,mask=bw)

            lanes_rgb,center_xy = self.postProcess(res)
            return lanes_rgb,center_xy

    def postProcess(self,image):
        src_img = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    
        red_mask = (src_img[:,:,2]>200).astype(np.uint8)
        src_img = cv2.bitwise_and(src_img,src_img,mask=1-red_mask)
        
        #Right Lanes
        green_mask = (src_img[:,:,1]>200).astype(np.uint8)
        green_area = cv2.bitwise_and(src_img,src_img,mask=green_mask)

        #Left Lanes
        blue_mask = (src_img[:,:,0]>200).astype(np.uint8)
        blue_area = cv2.bitwise_and(src_img,src_img,mask=blue_mask)

        lanes_rgb = cv2.addWeighted(green_area,1,blue_area,1,0)

        img_center_point,center_xy = self.window_search(green_mask,blue_mask)
        lanes_rgb = cv2.addWeighted(lanes_rgb,1,img_center_point,1,0)

        return lanes_rgb,center_xy

    @staticmethod
    def window_search(righ_lane, left_lane):
        center_coordinates =[]
        out = np.zeros(righ_lane.shape,np.uint8)
        out = cv2.merge((out,out,out))

        mid_point = np.int(righ_lane.shape[1]/2)

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

            histogram = np.sum(l_row, axis=0)
            l_point = np.argmax(histogram)

            if(l_point != 0) and (r_point != 0):
                rd = r_point-mid_point
                ld = mid_point-l_point
                if(abs(rd-ld)<100):
                    center = l_point + int((r_point-l_point)/2)
                    out = cv2.circle(out,(center,vp+win_y_center),2,(0,0,255),-1)
                    center_coordinates.append((center,vp+win_y_center))
        return out,center_coordinates