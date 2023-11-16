# 用于日常api、idea测试

import tensorflow as tf
import numpy as np


#基于tf的导向滤波
def guided_filter(x, y, r, eps=1e-2):

    def tf_box_filter(x, r):
        ch = x.get_shape().as_list()[-1]
        weight = 1 / ((2 * r + 1) ** 2)
        box_kernel = weight * np.ones((2 * r + 1, 2 * r + 1, ch, 1))
        box_kernel = np.array(box_kernel).astype(np.float32)
        output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
        return output

    x_shape = tf.shape(x)
    # y_shape = tf.shape(y)
    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output

if __name__ == '__main__':
    # a = tf.random.normal(shape=[1, 5, 5, 3])
    # kernel = tf.ones(shape=[1, 3, 3, 4])
    # b = tf.nn.depthwise_conv2d(a, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
    # print(b.shape)
    #
    # c = tf.nn.conv2d(a, filters=kernel, strides=[1, 1, 1, 1], padding="SAME")
    # print(c.shape)


    x = tf.random.normal(shape=[1, 5, 5, 3])
    out = guided_filter(x, x, 1)
    print(out.shape)

    # # numpy的卷积过程 卷积核需要翻转设计
    # y = np.convolve([1, 2, 3], [1, 0.5, 0], mode='same')
    # print(y)
    # # tf的卷积正常理解
    # inp = tf.cast([[[1, 2, 3]]],dtype=tf.float32)
    # print(inp.shape)
    # k = tf.cast([[[0],
    #               [0.5],
    #               [1]]], dtype=tf.float32)
    # print(k.shape)
    # y1 = tf.nn.conv1d(inp, filters=k, stride=[1, 1, 1], padding="SAME", data_format="NWC")
    # print(y1)
