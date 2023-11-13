import cv2
import numpy as np
from scipy import signal

# img_path = r'F:\data_analysis\illumination_contrast_balancing\band_3_elim.png'
img_path = r'F:\data_analysis\illumination_contrast_balancing\test2_elim.tif'
img_data_elim = cv2.imread(img_path, -1)

cv2.imshow('light_elim', img_data_elim)

h, w, c = img_data_elim.shape

blk_size = 61
epsilon = 1e-8

delta_x = img_data_elim[1:, :, :] - img_data_elim[:-1, :, :]  # [h-1, w, _]
delta_y = img_data_elim[:, 1:, :] - img_data_elim[:, :-1, :]  # [h, w-1, _]
residual = np.sqrt((delta_x[:, 1:w] ** 2 + delta_y[1:h, :] ** 2) / 2)  # [h-1, w-1, _]
print(residual.shape)
'''numpy自带卷积 只能1d'''
# defination_value = np.convolve(residual,
#                                np.ones(shape=(blk_size - 1, blk_size - 1)) / ((blk_size - 1) * (blk_size - 1)),
#                                mode='valid')

'''tf 卷积 调动tf包耗时'''
# conv_filter = np.ones(shape=(blk_size-1, blk_size-1, 3, 1)) / ((blk_size - 1) * (blk_size - 1))
# defination_value = tf.nn.conv2d(np.expand_dims(residual, axis=0), filters=conv_filter, strides=1, padding="VALID")

'''scipy 2d卷积'''
def_filter = np.ones(shape=(blk_size - 1, blk_size - 1), dtype=np.float32) / ((blk_size - 1) * (blk_size - 1))
defination_value = np.zeros(shape=((h - 1) - (blk_size - 1) + 1, (w - 1) - (blk_size - 1) + 1), dtype=np.float32)

nbr_filter = np.ones(shape=(blk_size, blk_size), dtype=np.float32)/(blk_size * blk_size)
Mean_nbr = np.zeros_like(img_data_elim, dtype=np.float32)
Std_nbr = np.zeros_like(img_data_elim, dtype=np.float32)

for ci in range(c):
    defination_value += signal.convolve2d(residual[..., ci], def_filter, mode='valid')
    Mean_nbr[..., ci] = signal.convolve2d(img_data_elim[..., ci], nbr_filter, mode='same', boundary='symm')
    Std_nbr[..., ci] = np.sqrt(signal.convolve2d((img_data_elim[..., ci] - Mean_nbr[..., ci])**2,
                                                 nbr_filter,
                                                 mode='same',
                                                 boundary='symm'))


ref_index = np.unravel_index(np.argmax(defination_value), defination_value.shape)  # 根据最大De值 找到指定ref_Block
print(ref_index)

ref_block = img_data_elim[ref_index[0]:ref_index[0] + blk_size, ref_index[1]:ref_index[1] + blk_size]
cv2.imshow('00', ref_block)

Mean_ref = np.mean(ref_block, axis=(0, 1))
Std_ref = np.std(ref_block, axis=(0, 1))

print(Mean_ref)
print(Std_ref)

"""逐像素计算 以blk_size为领域的 Mean & Std"""

print(Mean_nbr.shape)
print('Mean_nbr', np.mean(Mean_nbr, axis=(0, 1)))
print(Std_nbr.shape)
print('Std_nbr', np.mean(Std_nbr, axis=(0, 1)))

w_s = Std_ref / (Std_ref + Std_nbr)
w_m = Mean_ref / (Mean_ref + Mean_nbr)


alpha = w_s * Std_ref / (w_s * Std_nbr + (1 - w_s) * Std_ref)
beta = w_m * Mean_ref + (1 - w_m - alpha) * Mean_nbr

result = alpha * img_data_elim + beta
print(result.dtype, np.max(result), np.min(result))

result = (result-np.min(result))/(np.max(result)-np.min(result))
cv2.imshow('image_balancing', result)
cv2.waitKey()
