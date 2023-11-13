import cv2
import tifffile as tiff
import numpy as np
from osgeo import gdal

from labs.WangBo.make_dataset.utillity_MakeDataset import Data_Format_Conversion


# img_pth = r'F:\data_analysis\max_mean_min\test1.png'
# img_data = cv2.imread(img_pth, -1)

# img_pth = r"F:\Dataset_Satellite\ShangHai_16Bit\01_sources\0_img\RGB-PanSharpen_AOI_4_Shanghai_img121.tif"
# img_pth = r'\\192.168.1.53\qnap_ai\Xmap_Original_DATA\2022\20211228匀光匀色\06-07-1.tif'

# img_pth = r"F:\data_analysis\线性优化拉伸测试数据\cs1.tif"


# in_img_pth = r"F:\data_analysis\线性优化拉伸测试数据\cs2.JP2"
img_pth = r"F:\data_analysis\线性优化拉伸测试数据\cs2.tif"
# data_format_conversion = Data_Format_Conversion(in_img_pth, img_pth)
# data_format_conversion.write_img()
img_data = tiff.imread(img_pth)
img_data = img_data[..., :3]
# cv2.imshow('o', img_data)
# cv2.namedWindow('o', cv2.WINDOW_FREERATIO)
# cv2.resizeWindow('o', 1000, 750)
print(img_data.shape)

Min_Percent = 0.025
Max_Percent = 0.99

min_value = 0.
max_value = 1

Min_Adjust_Percent = 0.2
Max_Adjust_Percent = 0.7
bit = 16  # todo 影像位数
h, w, c = img_data.shape


# img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
# cv2.imshow('', img_gray)
# cv2.waitKey()
# print(img_gray)
# histogram = cv2.calcHist([img_gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])/(h * w)
# histogram = np.squeeze(histogram)  # [c, 256]
# print(histogram, histogram.shape)
histogram = np.zeros(shape=(c, 2**bit, 1))
channel = ['red', 'green', 'blue']
for ch, col in enumerate(channel):
    # 若直接使用clacHist()的accumulate=Ture的参数，会导致累计值超出计算机的储存范围
    histogram[ch] = cv2.calcHist([img_data], channels=[ch], mask=None, histSize=[2**bit], ranges=[0, 2**bit]) / (
            h * w)

histogram = np.squeeze(histogram)
print(histogram.shape)
cdf = np.zeros_like(histogram)
for x in range(2**bit):
    """累计分布;F(x) = P(X <= x)"""
    cdf[:, x:x+1] = np.sum(histogram[:, :x + 1], axis=1, keepdims=True)
    # dtype:float64; 服从U(0, 1)
# print(cdf, cdf.shape)

# a = [0, 0, 0]
# b = [0, 0, 0]
a = np.zeros(shape=3)
b = np.zeros(shape=3)
for ch, col in enumerate(channel):
    i = 0
    while cdf[ch, i] < Min_Percent:
        # print(cdf[0, a], Min_Percent)
        i += 1
    a[ch] = i
    print(a, cdf[ch, i])

    j = 2**bit-1
    while cdf[ch, j] > Max_Percent:
        # print(cdf[0, b], Max_Percent)
        j -= 1
    b[ch] = j
    print(b, cdf[ch, j])

c = a - Min_Adjust_Percent * (b - a)
d = b + Max_Adjust_Percent * (b - a)
print(c, d)

# 线性拉伸
img_correction = np.zeros_like(img_data, dtype=np.float32)
img_correction[np.where(img_data < c)] = min_value
img_correction[np.where(img_data > d)] = max_value

index = (img_data <= d) & (img_data >= c)
img_correction[index] = ((img_data*index - c*index)/((d-c)*index))[index]*(max_value-min_value)+min_value

print(np.min(img_correction), np.max(img_correction))

# cv2.imshow('Stretching', img_correction[..., ::-1])
# cv2.resizeWindow('Stretching', 1000, 750)
# cv2.namedWindow('Stretching', cv2.WINDOW_FREERATIO)
# cv2.imshow('', np.array(img_correction*255, dtype=np.uint8))
# cv2.waitKey()

out_pth = r'F:\data_analysis\线性优化拉伸测试数据\cs2_6.tif'
tiff.imwrite(out_pth, np.array(img_correction*255, dtype=np.uint8))
