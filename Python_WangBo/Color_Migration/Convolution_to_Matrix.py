
import cv2
import numpy as np


def convolution_by_matrix(input_data, kernel_size=(3, 5), stride=2, conv_model='mean'):
    """
    Parameters
    ----------
    input_data: 需要归一化到[0, 1],float
    kernel_size: (kernel_h, kernel_w) kernel的尺寸(h, w)
    stride: 卷积步长
    conv_model:卷积方式：
     'mean'：取均值 
     'min_pooling': 取最小值
     'max_pooling': 取最大值
    """

    h, w = input_data.shape[:2]
    input_data = np.reshape(input_data, newshape=(h, w, -1))  # 对多通道 单通道都适用
    h, w, c = input_data.shape

    if h % stride == 0:
        result_h = int(h / stride)
    else:
        result_h = int(h / stride) + 1

    if w % stride == 0:
        result_w = int(w / stride)
    else:
        result_w = int(w / stride) + 1      

    if conv_model == 'min_pooling':
        result = np.ones(shape=(result_h, result_w, c), dtype=np.float32)
    else:
        result = np.zeros(shape=(result_h, result_w, c), dtype=np.float32)
    
    padding_h = (result_h - 1) * stride + kernel_size[0] - h
    padding_w = (result_w - 1) * stride + kernel_size[1] - w

    padding_top = int(padding_h / 2)
    padding_bottom = padding_h - padding_top
    padding_left = int(padding_w / 2)
    padding_right = padding_w - padding_left
    print((padding_top, padding_bottom), (padding_left, padding_right))

    input_padding = np.pad(input_data, 
                           ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
                           mode='reflect')
    print('padding_shape:', input_padding.shape)

    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            # print(i*kernel_size[1]+j)
            # print(i,i+h, j,j+w)
            if conv_model == 'mean':
                # result = 初始化 zeros
                result += input_padding[i:i+h:stride, j:j+w:stride]
            if conv_model == 'min_pooling':
                # result = 初始化 ones
                result = np.minimum(result, input_padding[i:i+h:stride, j:j+w:stride])
            if conv_model == 'max_pooling':
                # result = 初始化 zeros
                result = np.maximum(result, input_padding[i:i+h:stride, j:j+w:stride])
    
    if conv_model == 'mean':
        result = result / (kernel_size[0] * kernel_size[1])

    return result


if __name__ == "__main__":
    img_path = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\img_balance_to_target\5m.tif"
    # img_path = fr'/Users/wangbo/PycharmProjects/dataset/img_balance_to_target/target.tif'
    img_data = cv2.imread(img_path, -1)
    import time

    # cv2.imshow('origin', img_data)
    # cv2.waitKey()
    print('img_data.shape', img_data.shape)
    k_size=[31, 31]
    t1 = time.time()
    conv_mean = convolution_by_matrix(img_data/255, kernel_size=k_size, stride=8, conv_model='mean')
    t2 = time.time()
    print(f'conv_by_mattrix 耗时：{t2-t1}', conv_mean.shape)
    cv2.imshow('conv_mean', conv_mean)
    cv2.waitKey()

    # 对比scipy的卷积
    from scipy import signal
    residual = img_data/255
    _, _, c = residual.shape
    t3 = time.time()
    scipy_mean = np.zeros_like(img_data, dtype=np.float32)
    filter = np.ones(shape=(k_size[0], k_size[1]), dtype=np.float32) / (k_size[0] * k_size[1])
    for ci in range(c):
        """逐像素计算 以blk_size为邻域的 Mean"""
        scipy_mean[..., ci] = signal.convolve2d(residual[..., ci], filter, mode='same', boundary='symm')
    t4 = time.time()
    print(f'scipy_conv 耗时：{t4-t3}', scipy_mean.shape)

    cv2.imshow('scipy_mean', scipy_mean)
    cv2.waitKey()
