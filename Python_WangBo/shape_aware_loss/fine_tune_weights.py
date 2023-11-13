# coding=utf-8
# 比较 两种shape weights 差异
import os
import sys

sys.path.append(os.path.abspath("../../../../"))

import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2 as cv
from scipy.ndimage import label as label_instances


# new， big会被强调, size_multiply 越小越强调, 待验证
def _get_size_weights_big(mask, size_multiply=4):
    C = np.sqrt(mask.shape[0] * mask.shape[1]) * size_multiply
    labels, num_labels = label_instances(mask, np.ones((3, 3)))
    sizes = np.ones_like(mask)
    for i in range(1, num_labels + 1):
        instance_size = (labels == i).sum()
        sizes = np.where(labels == i, instance_size, sizes)
    sizes_copy = sizes.copy()
    sizes_copy[sizes == 0] = 1
    weights = sizes_copy / C
    # all non-object areas get the weight of 1
    weights[sizes_copy == 1] = 1

    weights = np.where(weights < 1, 1., weights)  # leo added, 控制不要太大或太小
    weights = np.where(weights > 5, 5., weights)
    return weights.astype('float32')


# 缺省的，small会被强调
def _get_size_weights_small(mask, size_divosor=2.0):
    C = np.sqrt(mask.shape[0] * mask.shape[1]) / size_divosor
    labels, num_labels = label_instances(mask, np.ones((3, 3)))
    sizes = np.ones_like(mask)
    for i in range(1, num_labels + 1):
        instance_size = (labels == i).sum()
        print(f'instance_size:{instance_size}')

        # todo 2020.08.05 考虑到很小的instance对weight的量纲影响很大 对面积小于某阈值的instance 赋与一个固定值  WangBO
        if instance_size > 20 * 20:
            sizes = np.where(labels == i, instance_size, sizes)
        else:
            sizes = np.where(labels == i, 20 * 20, sizes)

    sizes_copy = sizes.copy()
    sizes_copy[sizes == 0] = 1  # 前景的值：instance_size[i] >=36 , 背景值为：1 原因：下一步取倒数，背景值不能为0
    weights = C / sizes_copy
    # all non-object areas get the weight of 1
    weights[sizes_copy == 1] = 0  # todo  2020.8.5 改为0， 原来是1， leo
    # weights = np.where(weights < 1, 1., weights)  # leo added, 控制不要太大或太小
    weights = np.sqrt(weights)  # todo 2020.08.05  考虑到面积之比是二次增长，用sqrt去除二次关系 WangBo
    print('weights', np.min(weights), np.max(weights))
    # weights = np.where(weights > 5, 5., weights)
    return weights.astype('float32')


def scale(image):
    i_max = image.max()
    i_min = image.min()
    smooth = 1e-12
    result = (image - i_min) / (i_max - i_min + smooth)

    return result * 1


def _compute_Edistance_weight2(mask_true):
    """
    找到 分割 与 ground truth 曲线周围点之间的欧式距离，并将其用作交叉熵损失函数的系数
    label.shape = [batch_size, h, w, c]
    """
    pos_mask = mask_true
    # print(mask_true.shape)  # (600, 600)
    neg_mask = 1 - mask_true
    Edistance_weights = np.zeros_like(mask_true, dtype=np.float32)

    Edistance = distance_transform_edt(pos_mask).astype(np.float32)
    max_Edistance = np.max(Edistance)  # 找到每张mask的最大距离
    weight = max_Edistance - Edistance  # 使得越靠近边缘的权重值越大
    weight = np.where(weight >= 1 * max_Edistance, 0, weight)  # 原本distance为0的依然保持为0
    weight = weight / (np.max(weight) + 1e-10)  # weight 归一化 smooth保证除数不为零
    Edistance_weights += weight

    Edistance2 = distance_transform_edt(neg_mask).astype(np.float32)
    max_Edistance2 = np.max(Edistance2)  # 找到每张mask的最大距离
    weight2 = max_Edistance2 - Edistance2  # 使得越靠近边缘的权重值越大
    weight2 = np.where(weight2 >= 1 * max_Edistance2, 0, weight2)  # 原本distance为0的依然保持为0
    weight2 = weight2 / (np.max(weight2) + 1e-10)  # weight 归一化 smooth保证除数不为零
    Edistance_weights += weight2

    return Edistance_weights


# 从边缘里外双向向边缘渐变
def _get_distance_weights_new(mask, w0=5.0, sigma=10.0):
    d = distance_transform_edt(1 - mask)
    weights = np.ones_like(mask) + w0 * np.exp(-(np.power(d, 2)) / (sigma ** 2))
    weights[d == 0] = 1

    d2 = distance_transform_edt(mask)
    weights2 = np.ones_like(1 - mask) + w0 * np.exp(-(np.power(d2, 2)) / ((sigma * 1) ** 2))
    weights2[d2 == 0] = 1
    return (weights + weights2).astype('float32')


# 从边缘向外部单向渐变
def _get_distance_weights(mask, w0=5.0, sigma=10.0):
    d = distance_transform_edt(1 - mask)
    weights = np.ones_like(mask) + w0 * np.exp(-(np.power(d, 2)) / (sigma ** 2))
    weights[d == 0] = 1

    return weights.astype('float32')


if __name__ == '__main__':
    # mask = cv.imread(r'E:\line\line_cut_labels\09FE083E_1_3.tif', -1)[:512, :512]
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv.dilate(mask, kernel, iterations=20) / 255  # 膨胀

    mask_dir = r'Z:\test_lbls'
    for files_name in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, files_name)
        mask = cv.imread(mask_path, -1)/255
        cv.imshow('mask', mask)
        print(f'mask, min={np.min(mask)},max={np.max(mask)}')

        # mask = cv.imread(r'W:\aas_dataset\02_sources\ancheng\cut_lbls\h29\AnCheng_1_7_6_4.tif', -1) / 255
        # print(f'mask, min={np.min(mask)},max={np.max(mask)}')

        weights_new = scale(_get_size_weights_small(mask)) * 2 + scale(_get_distance_weights_new(mask)) * 3
        print(f'weights_new, min={np.min(weights_new)},max={np.max(weights_new)}')

        weights_old = _get_distance_weights(mask)
        print(f'weights_old, min={np.min(weights_old)},max={np.max(weights_old)}')

        weights_lunwen = _compute_Edistance_weight2(mask)
        print(f'weights_lunwen, min={np.min(weights_lunwen)},max={np.max(weights_lunwen)}')

        cv.imshow('mask', mask)
        cv.imshow('weights_old', scale(weights_old))
        cv.imshow('weights_lunwen', scale(weights_lunwen))
        cv.imshow('weights_new', scale(weights_new))

        cv.waitKey()
        cv.destroyAllWindows()

        print('done.')
