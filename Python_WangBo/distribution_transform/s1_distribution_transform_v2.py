# encoding=utf-8
import os
import time
import cv2  # [BGR]
import tifffile as tiff  # [RGB]
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm


class DataDistributionTransform:
    """
    该类主要有一下几个功能：
    1. 对文件夹内图像像素X 进行频率统计，并绘制像素频率分布图
    2. 按照一定规律 对图像的像素X分布 变换为Y 服从均匀分布 即Y~U(0, 1)
    3. 在像素Y服从均匀分布的基础上 将数据进一步转换为Z服从高斯分布 即Z~N(0, 1)
    4. 分布回溯：Z~N(0, 1) --> Y~U(0, 1) --> X 即分布逆变换
    5. 模拟近似分布：数据样本X1服从分布F1, 数据样本X2服从分布F2,对X1进行一系列变换使得X1服从分布F2
    """

    def __init__(self, img_suffix=None, channel=None):
        """
        img_suffix: 图像文件的后缀 默认：['tif']
        channel: 图像文件的通道属性 默认：['red', 'green', 'blue']
        """
        if img_suffix is None:
            img_suffix = ['tif']
        self.img_suffix = img_suffix

        if channel is None:
            channel = ['red', 'green', 'blue']
        self.channel = channel

    def __getImgNameList(self, dataset_dir):
        """
        dataset_dir: 图像数据文件的夹路径

        return: 文件夹“dataset_dir”内图像数据的文件名 且满足后缀是['tif']格式的
        """
        return [name for name in os.listdir(dataset_dir) if name.split('.')[-1] in self.img_suffix]

    def __dataStatistics(self, img_data):
        """
        img_data: 图像数据，array，[H W C]

        return: 单张 img_data 直方统计 (像素比率 而不是像素个数 否则容易数值溢出)
        """
        h, w, c = img_data.shape
        histogram = np.zeros(shape=(c, 256, 1))
        for ch, col in enumerate(self.channel):
            # 若直接使用clacHist()的accumulate=Ture的参数，会导致累计值超出计算机的储存范围
            histogram[ch] = cv2.calcHist([img_data], channels=[ch], mask=None, histSize=[256], ranges=[0, 256]) / (
                    h * w)
        return histogram  # 返回值为 累积统计的频率

    def dataset_distribution(self, dataset_dir, picture=True):
        """
        dataset_dir: 图像数据文件的夹路径
        picture: 是否画统计图并保存（默认为True）

        对文件夹dataset_dir内的图像像素X 进行频率统计，并绘制像素频率分布图
        return: 累积统计的频率 shape:[chanel, 256]; 以及 像素分布体统计图
        """
        img_name_list = self.__getImgNameList(dataset_dir)
        """ 样本总体 pix的分布律 """
        histogram = np.zeros(shape=(len(self.channel), 256, 1))
        for i, img_dir in enumerate(img_name_list):
            t1 = time.time()
            '''读取原始数据 X = (x1, x2, x3) [R G B]'''
            img_data = cv2.imread(os.path.join(dataset_dir, img_dir))  # BGR
            # img_data = tiff.imread(os.path.join(dataset_dir, img_dir))  # [RGB]
            histogram = histogram + self.__dataStatistics(img_data)
            t2 = time.time()
            print(f'读取第{i + 1}张img, 读取时间：{t2 - t1}')

        histogram = np.squeeze(histogram / len(img_name_list))

        "绘制像素频率柱状图"
        if picture:
            dataset_name = dataset_dir.split('\\')[-1]
            plt.figure(figsize=(26, 10))
            plt.title(f"{dataset_name}'s Pixel Statistical Histogram", fontsize=20)
            plt.xlabel('pixle value', fontsize=18)
            plt.ylabel('pixle frequency', fontsize=18)
            plt.xlim([0, 256])
            width_val = (1 - 0.1) / len(self.channel)
            xticks = np.arange(256)
            for c in range(len(self.channel)):
                plt.bar(xticks + width_val * c, histogram[c], alpha=0.6, width=width_val, color=self.channel[c],
                        label=self.channel[c])

            hist_img_path = os.path.join(Path(dataset_dir).parent, dataset_name + '_distribution.jpg')
            plt.savefig(hist_img_path)
        return histogram

    def __cumulativeExperienceDistribution(self, dataset_dir, use_uniform01=True):
        """
        dataset_dir: 图像数据文件的夹路径
        use_uniform01: True:  dtype:'float64'; 服从U(0, 1);
                       False: dtype:'uint8'  ; 服从U(0,255)

        由经验分布 构造 均匀分布 随机变量：Xi --> Yi
        Yi = Fi(x) = P(X<=x) 累计经验分布函数 Fi(xi) 其中i = 1, 2, 3
        Yi ~ U(0, 1)

        return: 累计经验分布; shape:[3, 256]
        """
        histogram = self.dataset_distribution(dataset_dir, picture=False)
        cdf = np.zeros_like(histogram)
        for x in range(256):
            """累计分布;F(x) = P(X <= x)"""
            cdf[:, x:x + 1] = np.sum(histogram[:, :x + 1], axis=1, keepdims=True)
            # dtype:float64; 服从U(0, 1)
        if not use_uniform01:
            cdf = np.array(cdf * 255, dtype='uint8')
            # dtype:'uint8', 服从U(0,255)

        return cdf

    @staticmethod
    def __transfToUniform(origin_data, cdf):
        """
        origin_data: 图像数据，array，[H W C]
        cdf: 数据集的累计经验分布

        对数据集 datast 进行数据变换
        F：该数据集dataset的经验分布函数表

        """
        h, w, c = origin_data.shape

        uniform_data = np.array(np.zeros_like(origin_data), dtype='float32')
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    uniform_data[i, j, k] = cdf[k, origin_data[i, j, k]]  # 根据经验分布函数表 换算 对应的Y 其中yi ~ U[0, 1]

        # if use_uniform01:
        #     # 如果直接使用uniform_data cdf需要规范到0-255；如果是转normal_data的中间过程则不需要规范更好0-1
        #     uniform_data = np.array(uniform_data * 255, dtype='uint8')

        return uniform_data

    def originTransfToUniform(self, dataset_dir, use_uniform01=False):
        """
        dataset_dir: 图像数据文件的夹路径
        use_uniform01: True:  dtype:'float64'; 服从U(0, 1);
                       False: dtype:'uint8'  ; 服从U(0,255)

        按照一定规律 对图像的像素X分布 变换为Y 服从均匀分布 即Y~U(0, 1)

        """
        cdf = self.__cumulativeExperienceDistribution(dataset_dir, use_uniform01=use_uniform01)

        img_name_list = self.__getImgNameList(dataset_dir)
        for i, name in enumerate(img_name_list):
            t1 = time.time()

            # origin_data = tiff.imread(os.path.join(dataset_dir, name))  # [RGB]
            origin_data = cv2.imread(os.path.join(dataset_dir, name))  # [RGB]

            uniform_data = self.__transfToUniform(origin_data, cdf)

            dataset_name = dataset_dir.split('\\')[-1]
            uniform_save_dir = os.path.join(Path(dataset_dir).parent, f'{dataset_name}_to_Uniforms')
            if not os.path.exists(uniform_save_dir):
                os.makedirs(uniform_save_dir)
            uniform_save_name = os.path.join(uniform_save_dir, name)
            print("000", uniform_data.shape)
            tiff.imwrite(uniform_save_name, data=uniform_data)

            t2 = time.time()
            print(f'转换第{i + 1}张img, 耗时:{t2 - t1}')

    @staticmethod
    def __boxMullerTransform(Y1, Y2):
        """
        Y1, Y2: 服从均匀分布的随机的随机变量

        由均匀分布 构造 正态分布 随机变量：Y1 Y2 --> Z1 Z2
        z1 = sqr(-2*ln(y1)) * cos(2Π*y2)
        z2 = sqr(-2*ln(y1)) * sin(2Π*y3)

        return: z1, z2 ~ N2(0, 1), 即服从标准的高斯分布
        """
        R = np.sqrt(-2 * np.log(Y1))
        Z1 = R * np.cos(2 * np.pi * Y2)
        Z2 = R * np.sin(2 * np.pi * Y2)
        return Z1, Z2

    def __transfToNormal(self, uniform_data):
        """
        uniform_data： 均匀分布图像数据，array，[H W C]

        构造： Yi --> Zj  (i = 1 2 3) (j = 1 2 3 4 5 6) 其中Zj ~ N(0, 1)
        return: normal_data array,[H W C*2]
        """
        Z1, Z2 = self.__boxMullerTransform(uniform_data[..., 0:1], uniform_data[..., 1:2])
        Z3, Z4 = self.__boxMullerTransform(uniform_data[..., 1:2], uniform_data[..., 2:3])
        Z5, Z6 = self.__boxMullerTransform(uniform_data[..., 2:3], uniform_data[..., 0:1])
        normal_data = np.concatenate((Z1, Z2, Z3, Z4, Z5, Z6), axis=2)

        return normal_data

    def originTransfToStdNormal(self, dataset_dir):
        """
        dataset_dir: 图像数据文件的夹路径

        origin_data --> uniform_data --> normal_data
        在原像素X 转化为 像素Y 服从均匀分布的基础上 Y~U(0, 1); 将数据进一步转换为Z服从高斯分布 即Z~N(0, 1)
        """
        dataset_name = dataset_dir.split('\\')[-1]
        normal_save_dir = os.path.join(Path(dataset_dir).parent, f'{dataset_name}_to_Normals')
        if not os.path.exists(normal_save_dir):
            os.makedirs(normal_save_dir)

        uniform_name_list = self.__getImgNameList(dataset_dir)
        cdf = self.__cumulativeExperienceDistribution(dataset_dir, use_uniform01=True)
        for i, name in enumerate(uniform_name_list):
            t1 = time.time()

            origin_data = tiff.imread(os.path.join(dataset_dir, name))
            uniform_data = self.__transfToUniform(origin_data, cdf)
            normal_data = self.__transfToNormal(uniform_data)

            # cv2.imshow('012', normal_data[..., [0, 1, 2]])
            # cv2.imshow('345', normal_data[..., [0, 2, 4]])
            # cv2.waitKey()

            normal_save_name = os.path.join(normal_save_dir, name)
            tiff.imwrite(normal_save_name, data=normal_data)

            t2 = time.time()
            print(f'转换第{i + 1}张img, 耗时:{t2 - t1}')

    """逆变换"""

    @staticmethod
    def __boxMullerInversionTransform(Z1, Z2):
        """
        Z1 Z2: 服从N(0,1), 即服从高斯分布
        由高斯分布 逆变换 均匀分布：
        Y1 = arctan(Z1/Z2)/2Π  + 0.5
        Y2 = exp(-(Z1^2+Z2^2)/2)

        return: Y1, Y2: 服从U(0, 1), 即服从01均匀分布
        """
        Y1 = np.exp(-(Z1 ** 2 + Z2 ** 2) / 2)
        Y2 = np.arctan(Z1 / (Z2 + 1e-6)) / 2 * np.pi + 0.5
        return Y1, Y2

    def __stdNormalransfToUniform01(self, normal_data):  # chanel 为6
        """
        normal_data: 高斯分布图像数据，array，[H W C*2]
        构造： Zi --> Zj  Zi ~ N(0, 1)其中(i =0 1 2 3 4 5 ) ; Zj ~ U(0, 1)其中(j =0 1 2)
        return: uniform_data, array, [H W C]
        """
        Y1, Y2 = self.__boxMullerInversionTransform(normal_data[..., 0:1], normal_data[..., 1:2])
        Y3, Y4 = self.__boxMullerInversionTransform(normal_data[..., 2:3], normal_data[..., 3:4])
        Y5, Y6 = self.__boxMullerInversionTransform(normal_data[..., 4:5], normal_data[..., 5:6])
        return Y1, Y3, Y5

    def normalTransfToUiform(self, normaldata_dir, use_uniform01=True):
        """
        normaldata_dir: 高斯分布图像数据文件的夹路径
        use_uniform01: True:  dtype:'float64'; 服从U(0, 1);
                       False: dtype:'uint8'  ; 服从U(0,255)
        """
        img_name_list = self.__getImgNameList(normaldata_dir)
        for i, name in enumerate(img_name_list):
            t1 = time.time()

            normal_data = tiff.imread(os.path.join(normaldata_dir, name))
            print(normal_data.shape, normal_data.dtype)
            Y1, Y3, Y5 = self.__stdNormalransfToUniform01(normal_data)
            print(Y1.shape, Y1.dtype)
            uniform_data = np.concatenate((Y1, Y3, Y5), axis=2)

            if not use_uniform01:
                uniform_data = np.array(uniform_data * 255, dtype='uint8')

            t2 = time.time()
            print(f'转换时间:{t2 - t1}')
            dataset_name = normaldata_dir.split('\\')[-1]
            new_save_dir = os.path.join(Path(normaldata_dir).parent, f'{dataset_name}_to_uniform')
            if not os.path.exists(new_save_dir):
                os.makedirs(new_save_dir)

            new_save_name = os.path.join(new_save_dir, name)
            tiff.imwrite(new_save_name, data=uniform_data)

    def origin1transfToOrigin2(self, dataset_dir1, dataset_dir2):
        """
        dataset_dir1:
        dataset_dir2:
        近似分布: 数据样本X1服从分布F1, 数据样本X2服从分布F2, 对X1进行一系列变换使得X1服从分布F2
        """
        cdf2 = self.__cumulativeExperienceDistribution(dataset_dir2, use_uniform01=False)

        cdf1 = self.__cumulativeExperienceDistribution(dataset_dir1, use_uniform01=False)

        print('*'*100+'\n', cdf2, cdf1)

        img_name_list = self.__getImgNameList(dataset_dir1)
        for i, name in enumerate(img_name_list):
            t1 = time.time()
            img_data = tiff.imread(os.path.join(dataset_dir1, name))
            uniform_data = self.__transfToUniform(img_data, cdf1)
            img_new = np.array(np.zeros_like(uniform_data), dtype='uint8')
            H, W, C = uniform_data.shape
            print(H, W, C)
            for h in tqdm(range(H)):
                for w in range(W):
                    for c in range(C):

                        if uniform_data[h, w, c] in cdf2[c, :]:
                            index = np.where(cdf2[c, :, ] == uniform_data[h, w, c])[0]
                            # print('0', index)
                        else:
                            u_d_right = u_d_left = uniform_data[h, w, c]
                            while u_d_right not in cdf2[c, :] and u_d_left not in cdf2[c, :]:
                                # 如果累积分布表中不存在uniform[h,w,c]处的值，则通过其相邻值 在F中找index
                                u_d_right += 1
                                u_d_left -= 1
                            if u_d_right in cdf2[c, :]:
                                index = np.where(cdf2[c, :] == u_d_right)[0]
                                # print("1", index)
                            else:
                                index = np.where(cdf2[c, :] == u_d_left)[0]
                                # print("-1", index)
                        # print(index)
                        # print(np.mean(index, keepdims=False, dtype='uint8'))
                        # print(h, w, c)
                        img_new[h, w, c] = np.mean(index, keepdims=False, dtype='uint8')

            t2 = time.time()
            print(f'转换时间:{t2 - t1}')
            dataset_name1 = dataset_dir1.split('\\')[-1]
            dataset_name2 = dataset_dir2.split('\\')[-1]
            new_save_dir = os.path.join(Path(dataset_dir1).parent, f'{dataset_name1}_to_{dataset_name2}')
            if not os.path.exists(new_save_dir):
                os.makedirs(new_save_dir)

            new_save_name = os.path.join(new_save_dir, name)
            tiff.imwrite(new_save_name, data=img_new[..., ::-1])


if __name__ == '__main__':
    # """初始化一个实例对象"""
    data_distribute_transf = DataDistributionTransform()

    # # 1. 查看 data_dir 的像素频率统计图
    # dataset_dir01 = r'F:\data_distributed_transform\danyi\Image'  # itami城
    # hist = data_distribute_transf.dataset_distribution(dataset_dir01, picture=True)
    # print('像素比例分布表', hist)

    # # 2. 将 data_dir 转换为 均匀分布数据
    # dataset_dir01 = r'F:\data_distributed_transform\danyi\Image'  # itami城
    # data_distribute_transf.originTransfToUniform(dataset_dir01)

    # # 3. 将 data_dir 转换为 高斯分布数据
    # dataset_dir01 = r'F:\data_distributed_transform\itamiH29\Images'  # itami城
    # data_distribute_transf.originTransfToStdNormal(dataset_dir01)

    # # 4. 将 高斯分布 转换为 均匀分布
    # normaldata_dir1 = r'F:\data_distributed_transform\itamiH29\Images_to_Normals'
    # data_distribute_transf.normalTransfToUiform(normaldata_dir1, use_uniform01=False)

    # 5. 使得dataset_dir1 转换为 dataset_dir2的分布
    dataset_dir01 = r'F:\data_analysis\distributed_transform\itamiH29\Images_0.2'  # itami城
    dataset_dir02 = r'F:\data_analysis\distributed_transform\ancheng\h29_0.05'  # 安城
    data_distribute_transf.origin1transfToOrigin2(dataset_dir01, dataset_dir02)

    # data_distribute_transf = DataDistributionTransform(img_suffix='jpg')
    # dataset_dir03 = r'F:\data_analysis\Optimized_Linear_Stretch\test_real'
    # data_distribute_transf.originTransfToUniform(dataset_dir03, use_uniform01=True)
