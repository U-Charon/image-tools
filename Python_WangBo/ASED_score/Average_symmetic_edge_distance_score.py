import os

import cv2
import numpy as np
import tifffile


# print(tifffile.__version__)
# print(cv2.__version__)


def find_contours(img_path):
    '''
    寻找二值图的边界点集

    return：
       contour_img：边界二值图；
       contours_set：边界点集
    '''
    img = tifffile.imread(img_path)

    contours_img = np.zeros_like(img)
    contours_set = []
    w, h = img.shape
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            if img[i, j] != 0:
                if img[i - 1, j] == 0 or img[i + 1, j] == 0 or img[i, j - 1] == 0 or img[i, j + 1] == 0:
                    contours_img[i, j] = 255  # 满足条件时，[i,j]位置即为边界点
                    contours_set.append((i, j))  # 点集集合

    return contours_img, contours_set


def Average_Symmtic_Edge_Distance(g_set, p_set, alpha=1):
    '''
    g_set: GroundTruth 的边界点集
    p_set: Prediction 的边界点集
    alpha: 调节整体分数值score对distance的灵敏度，alpha∈[0, 1]，默认值1，alpha越小，灵敏度越弱，区别区间越大

    return:
       distance_mat: GroundTruth各点 到 Prediction各点 的距离，一般不输出
       distance_score: GroundTruth 与 Prediction 差异分数，差异越大，值越小
    '''
    g_n = len(g_set)
    p_m = len(p_set)

    '''计算距离矩阵'''
    distance_mat = np.zeros(shape=(g_n, p_m))
    print(distance_mat.shape)
    for i, g_point in enumerate(g_set):
        for j, p_point in enumerate(p_set):
            distance = np.sqrt((g_point[0] - p_point[0]) ** 2 + (g_point[1] - p_point[1]) ** 2)
            distance_mat[i, j] = distance

    '''计算GroundTruth 到 Prediction 的 distance'''
    g_point_to_p = np.min(distance_mat, axis=1)  # 对j列取min 即为 g_i 到 p 的距离
    print(len(g_point_to_p), g_n, '************')
    g_sort_dist = np.sort(g_point_to_p)[:int(g_n * 0.95)]  # 排序取前95%的dist，剔除异常值，使得dist值更稳定
    g_to_p_dist = np.mean(np.exp(-alpha * g_sort_dist))  # 使用指数 exp(-dist) 将dist∈[0,+∞] 转化为score∈[0,1]
    # g_to_p_dist = np.mean(1 / (1 + alpha * g_sort_dist))  # 使用 1/(1+dist) 将dist转化为score

    print(g_to_p_dist, 'g_to_p_dist ')

    '''计算Prediction 到 GroundTruth 的 distance'''
    p_point_to_g = np.min(distance_mat, axis=0)  # 对i行取min 即为 p_j 到 g 的距离
    p_sort_dist = np.sort(p_point_to_g)[:int(p_m * 0.95)]  # 排序取前95%的dist，剔除异常值，使得dist值更稳定
    p_to_g_dist = np.mean(np.exp(-alpha * p_sort_dist))
    # p_to_g_dist = np.mean(1 / (1 + alpha * p_sort_dist))
    print(p_to_g_dist, 'p_to_g_dist')

    distance_score = 0.5 * (g_to_p_dist + p_to_g_dist)
    return distance_mat, distance_score


if __name__ == '__main__':

    gt_dir = r'special_dataset/gt'
    pd_dir = r'special_dataset/pd'

    # gt_dir = r'Z:\AI评价指标样本数据\gt'
    # pd_dir = r'Z:\AI评价指标样本数据\pred'

    score = 0
    n = 0
    for filename in os.listdir(gt_dir):
        if filename.split('.')[-1] == 'tif':
            print(filename)
            gt_path = os.path.join(gt_dir, filename)
            pd_path = os.path.join(pd_dir, filename)
            print(gt_path, pd_path)

            g_contours, g_set = find_contours(gt_path)
            cv2.imshow('groundtruth_contours', mat=g_contours)
            print(g_set)

            p_contours, p_set = find_contours(pd_path)
            cv2.imshow('pred_contours', mat=p_contours)
            print(p_set)

            cv2.imshow('-', mat=np.abs(g_contours - p_contours))

            distance_mat, distance_score = Average_Symmtic_Edge_Distance(g_set, p_set, alpha=0.5)
            score += distance_score
            n += 1
            print(distance_mat)
            print(distance_score)
            print('all done')
            # cv2.waitKey()

    avg_score = score / n
    print(avg_score)
