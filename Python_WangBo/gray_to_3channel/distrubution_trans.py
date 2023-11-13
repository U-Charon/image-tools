import os
import time

import cv2  # 大型数据读取很慢
import joblib
import matplotlib.pyplot as plt
import numpy as np

"""
数据的预处理
"""


class Data_Distribute:

    def __init__(self, dataset_dir, save_dir):
        """
        dataset_dir: 原数据文件地址
        save_dir: 转换后的数据文件存储地址
        """
        self.dataset_dir = dataset_dir
        self.img_listdir = os.listdir(self.dataset_dir)
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.hist = np.zeros(shape=[1, 256, 1])  # 单通道的像素值统计
        self.F = np.zeros_like(self.hist)  # 经验分布表
        self.colors = ['gray']  # 灰度图

    def __data_statistics(self, img_data, hist=np.zeros(shape=(1, 256, 1))):
        """
        对每一张 img_data 直方统计并累计
        """
        h, w = img_data.shape
        for ch, col in enumerate(self.colors):
            hist[ch] = (hist[ch] + cv2.calcHist([img_data], channels=[ch], mask=None, histSize=[256],
                                                ranges=[0, 255]) / (h * w))
            # 实时 统计变化
            plt.plot(hist[ch])
            plt.show()
            plt.pause(0.1)
        return hist  # 返回值为 累积统计的频率

    def __experience_distribution(self):
        """
        样本总体 pix的分布律
        """
        plt.ion()
        for i, img_dir in enumerate(self.img_listdir):
            print(f'读取第{i + 1}张img')
            t1 = time.time()
            '''读取原始数据 X = (x1, x2, x3) [R G B]'''
            img_data = cv2.imread(os.path.join(self.dataset_dir, img_dir))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)  # 转灰度图
            self.hist = self.__data_statistics(img_data, self.hist)
            t2 = time.time()
            print(f'读取时间：{t2 - t1}')
        plt.ioff()
        plt.close()
        print('num_image:', i+1)
        self.hist = self.hist/(i+1)

        plt.plot(self.hist[0], self.colors[0])  # 打印gray通道 pix统计折线图
        plt.savefig('./' + self.dataset_dir.split('\\')[-1] + '_distribution.jpg')
        return self.hist

    def __cdf_func(self, hist, x):
        """
        F(x) = P(X <= x)
        """
        return np.sum(hist[:, :x + 1, :], axis=1, keepdims=True)

    def experience_distribution_fit(self):
        """
        拟合一个分布函数 Fi(xi) 其中i = 1, 2, 3
        并返回一个经验分布表self.F
        """

        hist = self.__experience_distribution()

        for x in range(256):
            self.F[:, x:x + 1, :] = self.__cdf_func(hist, x)
            print(self.F[:, x:x + 1, :])
        print(f'经验分布表的shape：{self.F.shape}')
        joblib.dump(self.F, os.path.join(self.save_dir, 'experience_dist_table'))
        return self.F

    def __box_muller_transform(self, Y, dim):  # dim = 0,1,2
        """
        由均匀分布 构造 正态分布 随机变量：Y1 Y2 --> Z1 Z2
        z1 = sqr(-2*ln(y1)) * cos(2Π*y2)
        z2 = sqr(-2*ln(y1)) * sin(2Π*y3)
        Z = (z1, z2) ~ N2(0, 1), 即服从标准的高斯分布
        """
        R = np.sqrt(-2 * np.log(Y[:, :]))
        Z1 = R * np.cos(2 * np.pi * Y[:, :])
        Z2 = R * np.sin(2 * np.pi * Y[:, :])

        return Z1, Z2

    def distribute_transf(self):
        """
        对数据集 datast 进行数据变换
        F：该数据集dataset的经验分布函数表
        """
        F = self.experience_distribution_fit()
        for n, img_dir in enumerate(self.img_listdir):
            print(f'第{n + 1}张图开始转换：')
            t1 = time.time()

            img_data = cv2.imread(os.path.join(self.dataset_dir, img_dir))  # [600 600 3]
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)  # [600 600]

            (h, w), c = img_data.shape, 1
            img_uniform = np.array(np.zeros_like(img_data), dtype=np.float)  # [600 600]

            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        img_uniform[i, j] = F[k, img_data[i, j], 0]  # 根据经验分布函数表 换算 对应的Y 其中yi ~ U[0, 1]

            print(img_uniform.shape)

            '''构造： Yi --> Zj  (i = 1 2 3) (j = 1 2 3 4 5 6) 其中Zj ~ N(0, 1)'''
            Z1, Z2 = self.__box_muller_transform(img_uniform, dim=1)  # [600 600]
            img_normal = np.expand_dims((Z1+Z2)/2, axis=2)  # [600 600 1]
            result = np.zeros(img_normal.shape, dtype=np.float32)
            cv2.normalize(img_normal, result, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # 归一化处理
            img_normal = np.uint8(result * 255.0)
            # cv2.imshow("normal", img_normal)
            # print('max', np.max(img_normal), 'min', np.min(img_normal))
            '''normal_data 的数据保存'''
            # joblib.dump(img_normal, os.path.join(self.normal_path, img_dir.split('.')[0]))  # 存储正态分布的数据

            img_uniform = np.expand_dims(img_uniform, axis=2)
            img_uniform = np.uint8(img_uniform * 255)
            # cv2.imshow('uniform', img_uniform)  # todo 保存的数据需要按照imshow的格式保存
            # print('max', np.max(img_uniform), 'min', np.min(img_uniform))
            '''uniform_data 的数据保存'''
            # joblib.dump(img_uniform, os.path.join(self.uniform_path, img_dir.split('.')[0]))  # 存储均匀分布数据

            img_data = np.expand_dims(img_data, axis=2)
            # result = np.zeros(img_data.shape, dtype=np.float32)
            # cv2.normalize(img_data, result, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # 原图在这不用考虑添加normalize
            img_data = np.uint8(img_data * 255.0)
            # cv2.imshow('o', img_data)
            # print('max', np.max(img_data), 'min', np.min(img_data))

            img = np.concatenate((img_data, img_uniform, img_normal), axis=2)  # [600 600 3]
            print(img.shape)
            # cv2.imshow('img', img)
            cv2.imwrite(filename=os.path.join(self.save_dir, img_dir.split('.')[0]+'.png'), img=img)
            # cv2.waitKey()

            t2 = time.time()
            print(f'转换时间:{t2 - t1}')


if __name__ == '__main__':

    dataset_dir = r'G:\ailab\labs\WangBo\work2_data_analysis\data_samplesets'
    save_dir = r'G:\ailab\labs\WangBo\work2_data_analysis\gray_to_3channel\data_save'

    data_distribution = Data_Distribute(dataset_dir=dataset_dir, save_dir=save_dir)

    data_distribution.distribute_transf()

    print('all done')
