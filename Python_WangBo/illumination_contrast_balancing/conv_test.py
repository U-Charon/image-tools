import numpy as np
from scipy import signal
import time
import cv2

import os
vipshome = r'G:\SoftwarePackage\DeepLearning_Package\vips-dev-w64-web-8.12.2\vips-dev-8.12\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips

img_pth = r"F:\data_analysis\illumination_contrast_balancing\max_mean_min\test1.png"
# img_data = cv2.imread(img_pth, -1)
# # img_data = np.random.randint(0, 255, (2465, 1724, 3))
#
# blk_size = 9
# nbr_filter = np.ones(shape=(blk_size, blk_size), dtype=np.float32) / (blk_size ** 2)
#
# t1 = time.time()
# for i in range(3):
#     t1 = time.time()
#     mean_nbr = signal.convolve2d(img_data[..., i], nbr_filter, mode='same', boundary='symm')
#     t2 = time.time()
#     print(t2 - t1)
#     print(mean_nbr.shape)
#     std_nbr = np.sqrt(signal.convolve2d((img_data[..., i] - mean_nbr) ** 2,
#                                                  nbr_filter,
#                                                  mode='same',
#                                                  boundary='symm'))

im_data = pyvips.Image.new_from_file(img_pth)



