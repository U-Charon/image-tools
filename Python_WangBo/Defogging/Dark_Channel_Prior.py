import numpy as np
import cv2
from pathlib import Path
from Python_WangBo.Color_Migration.Convolution_to_Matrix import convolution_by_matrix



def defogging(img_pth, out_pth):

    img_data = cv2.imread(img_pth, -1)
    print(img_data.shape)
    h, w, c = img_data.shape
    img_data = cv2.resize(img_data, (int(w/10), int(h/10)))
    print(img_data.min(), img_data.max(), img_data.shape)
    # cv2.namedWindow('0', cv2.WINDOW_FREERATIO)
    cv2.imshow('0', img_data)


    # for i in range(c):
    #     dark_i = convolution_by_matrix(img_data[..., i], kernel_size=[41, 41], stride=1, conv_model='min_pooling')
    #     print(dark_i.min(), dark_i.max())
    #     # cv2.namedWindow('dark_i', cv2.WINDOW_FREERATIO)
    #     cv2.imshow('dark_i', dark_i)
    #     A_i = convolution_by_matrix(dark_i, kernel_size=[41, 41], stride=1, conv_model='max_pooling')
    #     print(A_i.min(), A_i.max())
    #     # cv2.namedWindow('A_i', cv2.WINDOW_FREERATIO)
    #     cv2.imshow('A_i', A_i)
    #     cv2.waitKey()

    # 暗通道min滤波
    dark = convolution_by_matrix(img_data, kernel_size=[3, 3], stride=1, conv_model='min_pooling')
    print('dark', dark.min(), dark.max(), dark.shape)
    # # cv2.namedWindow('dark_i', cv2.WINDOW_FREERATIO)
    # cv2.imshow('dark', dark.astype(np.uint8))

    # 估算局部大气光值A
    # A = convolution_by_matrix(dark, kernel_size=[873, 1164], stride=1, conv_model='max_pooling')  # 局部大气光值
    # print('A', A.min(), A.max(), A.shape)
    # cv2.namedWindow('A', cv2.WINDOW_FREERATIO)
    # cv2.imshow('A_r', A[..., 0].astype(np.uint8))
    # cv2.imshow('A_g', A[..., 1].astype(np.uint8))
    # cv2.imshow('A_b', A[..., 2].astype(np.uint8))
    # cv2.imshow('A', A.astype(np.uint8))

    # max 估算大气光值
    A = np.max(dark, axis=(0, 1))
    print("A值:", A)

    # 计算transmission
    # print(img_data/A)
    transmission = 1 - 0.85*convolution_by_matrix(img_data/A, kernel_size=[31, 31], stride=1, conv_model='min_pooling')

    print('transmission', transmission.min(), transmission.max(), transmission.shape)
    # transmission =transmission
    # # cv2.imshow('t_r', transmission[..., 0].astype(np.uint8))
    # # cv2.imshow('t_g', transmission[..., 1].astype(np.uint8))
    # # cv2.imshow('t_b', transmission[..., 2].astype(np.uint8))
    # cv2.imshow('transmission', transmission.astype(np.uint8))
    # # cv2.waitKey()

    out = (img_data - A)/(transmission) + A
    print('out', out.min(), out.max(), out.shape)
    out = (out - out.min())/(out.max()-out.min()) *255

    cv2.imshow('out', out.astype(np.uint8))
    cv2.waitKey()
    # cv2.imwrite(out_pth, out.astype(np.uint8))

if __name__ == '__main__':
    # img_pth = r'.\data_test\need_tune\1016-PM-C1-DSCF5538.jpg'
    # img_data = cv2.imread(img_pth)
    #
    # out_pth = "./data_test/5619_new.jpg"
    # defogging(img_pth, out_pth)

    # img_dir = Path(r'.\data_test\need_tune')
    # for img_pth in img_dir.glob("*.jpg"):
    #     print(img_pth)
    #     defogging(str(img_pth), '')

    import tensorflow as tf
    a = tf.random.normal(shape=[1, 5, 5, 3])
    kernel = tf.ones(shape=[1, 3, 3, 4])
    b = tf.nn.depthwise_conv2d(a, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
    print(b.shape)

    c = tf.nn.conv2d(a, filters=kernel, strides=[1, 1, 1, 1], padding="SAME")
    print(c.shape)