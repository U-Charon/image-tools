import os
import sys

sys.path.append(os.path.abspath('../../../../'))  # home/ai/XmapAI

import cv2
import tifffile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __name__ == '__main__':
    '''采样转换'''
    # # dataset_dir = r'Z:\city1_kangzhou'
    # dataset_dir = r'./data_testsets'

    # dataset_dir = r'data_samplesets'  # 采样后的数据[600 600 3
    # save_dir = r'./data_save'

    # data_distribution = Data_Distribute(dataset_dir=dataset_dir, save_dir=save_dir)
    # data_distribution.distribute_transf()

    original_dir = r'F:\ass_dataset\02_sources\ancheng_mini\cut_imgs\h29'
    uniform_dir = r'F:\ass_dataset\02_sources\ancheng_mini\cut_imgs\h29_distribution_transform\uniform_data'
    normal_dir = r'F:\ass_dataset\02_sources\ancheng_mini\cut_imgs\h29_distribution_transform\normal_data'

    # dataset_dir = r'F:\ancheng\00_img\h30'
    # save_dir = r'F:\ancheng\00_img\transform_h30'

    # data_distribution = Data_Distribute(dataset_dir=dataset_dir, save_dir=save_dir)
    # data_distribution.distribute_transf()

    '''查看结果'''
    # original_dir = r'F:\ancheng\00_img\h30'
    # uniform_dir = r'F:\ancheng\00_img\transform_h30\uniform_data'
    # normal_dir = r'F:\ancheng\00_img\transform_h30\normal_data'

    original_list = os.listdir(original_dir)
    uniform_list = os.listdir(uniform_dir)
    normal_list = os.listdir(normal_dir)

    colors = ['blue', 'green', 'red']

    for i in range(len(original_list)):
        print(f'展示第{i + 1}张')
        original_data = cv2.imread(os.path.join(original_dir, original_list[i]))
        print(os.path.join(original_dir, original_list[i]))
        print(type(original_data))
        print(original_data.shape)
        cv2.namedWindow('original_img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('original_img', original_data)

        # uniform_data = joblib.load(os.path.join(uniform_dir, uniform_list[i]))
        uniform_data = tifffile.imread(files=os.path.join(uniform_dir, uniform_list[i]))
        cv2.imshow('uniform_img', uniform_data)

        # normal_data = joblib.load(os.path.join(normal_dir, normal_list[i]))
        normal_data = tifffile.imread(files=os.path.join(normal_dir, normal_list[i]))
        cv2.imshow('normal_img_1_3', normal_data[:, :, :3])
        cv2.imshow('normal_img_3_6', normal_data[:, :, 3:6])

        # cv2.waitKey()
        h, w, c = np.shape(original_data)

        plt.subplot(221)
        plt.title('original_hist')
        for ch, col in enumerate(colors):
            hist = cv2.calcHist([original_data], channels=[ch], mask=None, histSize=[256], ranges=[0, 256]) / (
                    h * w)
            plt.plot(hist, col)

        plt.subplot(222)
        plt.title('uniform_hist')
        uniform_data = Image.fromarray((255 * uniform_data).astype(np.uint8))
        uniform_data = cv2.cvtColor(np.asarray(uniform_data), cv2.COLOR_RGB2BGR)
        # cv2.imshow('cv2_uniform', uniform_data)
        # cv2.waitKey()
        for ch, col in enumerate(colors):
            hist = cv2.calcHist([uniform_data], channels=[ch], mask=None, histSize=[256], ranges=[0, 256]) / (h * w)
            plt.plot(hist, col)

        min_normal = np.min(normal_data, axis=(0, 1))
        max_normal = np.max(normal_data, axis=(0, 1))
        normal_data = (normal_data - min_normal) / (max_normal - min_normal) * 255

        plt.subplot(223)
        plt.title('normal_hist0-2')
        normal_data_0to2 = Image.fromarray((normal_data[:, :, :3]).astype(np.uint8))
        normal_data_0to2 = cv2.cvtColor(np.asarray(normal_data_0to2), cv2.COLOR_RGB2BGR)
        # cv2.imshow('cv2_normal_0to2', normal_data_0to2)
        # cv2.waitKey()
        for ch, col in enumerate(colors):
            hist = cv2.calcHist([normal_data_0to2], channels=[ch], mask=None, histSize=[256], ranges=[0, 256]) / (
                    h * w)
            plt.plot(hist, col)

        plt.subplot(224)
        plt.title('normal_hist_4-6')
        normal_data_3to5 = Image.fromarray((normal_data[:, :, 3:6]).astype(np.uint8))
        normal_data_3to5 = cv2.cvtColor(np.asarray(normal_data_3to5), cv2.COLOR_RGB2BGR)
        # cv2.imshow('cv2_normal_3to5', normal_data_3to5)
        cv2.waitKey()
        for ch, col in enumerate(colors):
            hist = cv2.calcHist([normal_data_3to5], channels=[ch], mask=None, histSize=[256], ranges=[0, 256]) / (
                    h * w)
            plt.plot(hist, col)
        plt.show()

    print('all done')