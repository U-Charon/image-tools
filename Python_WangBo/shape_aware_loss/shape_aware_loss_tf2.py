import time

import tensorflow as tf
from scipy import ndimage


def compute_Edistance_weight(label):
    """
    找到 分割 与 ground truth 曲线周围点之间的欧式距离，并将其用作交叉熵损失函数的系数
    label.shape = [batch_size, h, w, c]
    """
    pos_mask = label
    neg_mask = 1 - label
    # mask = tf.stack((pos_mask, neg_mask), axis=0)
    # print(mask.shape)
    weights = tf.zeros_like(label, dtype=tf.float32)

    for i, mask in enumerate([pos_mask, neg_mask]):
        Edistance = ndimage.distance_transform_edt(mask)  # 计算图像中非零点（即pix = 1的点）到 最近 背景点（即pix = 0的点）的欧式距离
        # print(Edistance.shape)
        Edistance = tf.constant(Edistance, dtype=tf.float32)
        max_Edistance = tf.reduce_max(Edistance, axis=[1, 2, 3], keepdims=True)  # 找到每张mask的最大距离
        weight = max_Edistance - Edistance  # 使得越靠近边缘的权重值越大
        weight = tf.where(weight >= 1 * max_Edistance, 0, weight)  # 原本distance为0的依然保持为0
        weight = weight / (tf.reduce_max(weight, axis=[1, 2, 3], keepdims=True) + 1e-10)  # weight 归一化 smooth保证除数不为零
        weights = tf.add(weights, weight)
    return weights


def shape_aware_Loss(net_output, label, Edistance_weights, smooth=1e-5):
    '''
    net_output: [batch_size 512 512 1]
    label: [batch_size 512 512 1]
    smooth：weight==0时会损失信息，故考虑加上很小的值
    '''
    bceLoss = tf.keras.backend.binary_crossentropy(label, net_output)
    shape_aware_Loss = (Edistance_weights + smooth) * bceLoss
    return shape_aware_Loss


if __name__ == '__main__':

    # net_output = tf.random.normal(shape=(2, 512, 512, 1), dtype=tf.float32)
    # label = tf.cast(tf.where(net_output >= 0, 1, 0), dtype=tf.float32)

    net_output = [[[[0.10], [0.20], [0.21], [0.32], [0.11], [0.11]],
                   [[0.22], [0.31], [0.81], [0.92], [0.20], [0.10]],
                   [[0.11], [0.33], [0.91], [0.86], [0.96], [0.12]],
                   [[0.22], [0.95], [0.86], [0.98], [0.88], [0.13]],
                   [[0.11], [0.03], [0.79], [0.86], [0.02], [0.23]],
                   [[0.16], [0.32], [0.21], [0.24], [0.14], [0.21]]],
                  [[[0.10], [0.20], [0.21], [0.32], [0.11], [0.11]],
                   [[0.22], [0.31], [0.81], [0.92], [0.20], [0.10]],
                   [[0.11], [0.33], [0.91], [0.86], [0.96], [0.12]],
                   [[0.22], [0.95], [0.86], [0.98], [0.88], [0.13]],
                   [[0.11], [0.03], [0.79], [0.86], [0.02], [0.23]],
                   [[0.16], [0.32], [0.21], [0.24], [0.14], [0.91]]]]

    net_output = tf.constant(net_output, dtype=tf.float32)
    label = tf.round(net_output)

    print(tf.squeeze(label))
    print(tf.squeeze(tf.slice(label, [0, 1, 1, 0], [2, 2, 2, 1])))

    t1 = time.time()
    Edistance_weights = compute_Edistance_weight(label)
    print(tf.squeeze(Edistance_weights))
    loss = shape_aware_Loss(net_output, label, Edistance_weights)
    t2 = time.time()
    print(tf.reduce_mean(loss, axis=[1, 2, 3]), f'time: {t2 - t1}')