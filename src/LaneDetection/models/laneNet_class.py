import cv2
import numpy as np
import tensorflow as tf
import os
import sys

r_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(r_path,"LaneNet"))

import lanenet
#from lanenet_model import lanenet_postprocess
import parse_config_utils


#CFG = parse_config_utils.lanenet_cfg
#LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


class LaneNet(object):
    def __init__(self,weights_path):
        self.cfg = parse_config_utils.lanenet_cfg
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.net = lanenet.LaneNet(phase='test', cfg=self.cfg)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')

        #self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=self.cfg,
        #ipm_remap_file_path=r"/home/eren/Documents/Carla/carla-ros-bridge/catkin_ws/src/ros-bridge/Carla_Eren/src/LaneKeepingAsistant/laneNet/data/tusimple_ipm_remap.yml")
        
        self.weights_path = weights_path

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

        # define saver
        self.saver = tf.train.Saver(variables_to_restore)



    @staticmethod
    def preProcessing(image):
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        return image



    def predict(self,image):
        src_image = self.preProcessing(image)
        
        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=self.weights_path)
            self.binary_seg_image, self.instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [src_image]}
            )
            """
            postprocess_result = self.postprocessor.postprocess(
                binary_seg_result=self.binary_seg_image[0],
                instance_seg_result=self.instance_seg_image[0],
                source_image=image
            )
            """
            rgb = self.instance_seg_image[0].astype(np.uint8)
            bw = self.binary_seg_image[0].astype(np.uint8)
            res = cv2.bitwise_and(rgb,rgb,mask=bw)
            return res

if __name__ == '__main__':
    import cv2
    import numpy as np

    image_path = r"/home/eren/Codes/Python/LaneNet/lanenet-lane-detection/data/tusimple_test_image/0.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ln = LaneNet(r"weights/tusimple_lanenet.ckpt")
    
    res = ln.predict(image)

    cv2.imshow("res",res.astype(np.uint8))
    cv2.waitKey(0)
