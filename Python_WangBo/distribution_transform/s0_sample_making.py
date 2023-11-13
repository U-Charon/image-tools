import os
import time

import cv2
# from PIL import Image
import numpy as np

''''''
'''采集样本 尺寸：[600, 600]'''
n = 30  # 增样倍数

data_dir = r'../data_testsets'

#  储存样本路径
save_path = r'../data_samplesets'

pic_size = 600
print('Sample_size:{}'.format(pic_size))
start = time.process_time()

if not os.path.exists(save_path):
    os.makedirs(save_path)

for files_name in os.listdir(data_dir):

    # shadow_img 和shadow_free 的真实路径
    img_path = os.path.join(data_dir, files_name)

    f_str = files_name.split('.')

    #  打开图片
    img = cv2.imread(img_path)
    print(img.shape)

    #  获得pic的宽高
    img_w, img_h, img_ch = img.shape

    '''裁剪n次'''
    for _n in range(n):
        #  在一定范围呢 选择中心区域
        crop_center_x = np.random.randint(pic_size / 2, img_w - pic_size / 2)
        crop_center_y = np.random.randint(pic_size / 2, img_h - pic_size / 2)

        box_x1 = crop_center_x - pic_size / 2
        box_y1 = crop_center_y - pic_size / 2
        box_x2 = crop_center_x + pic_size / 2
        box_y2 = crop_center_y + pic_size / 2
        print(box_x1, box_x2, box_y1, box_y2)

        img_crop = img[int(box_x1):int(box_x2), int(box_y1):int(box_y2), :]
        cv2.imwrite(os.path.join(save_path, f_str[0] + f'_{_n}.tif'), img_crop)
        # img_crop.imwrite(os.path.join(data_dir, f_str[0]+f'_{_n}.jpg'))

        # shadow_img_crop.show()

end = time.process_time()
print('时间：{0}s'.format(end - start))

# Sample_size:64
# 时间：37.515625s
# Sample_size:128
# 时间：40.9375s
# Sample_size:224
# 时间：52.953125s
