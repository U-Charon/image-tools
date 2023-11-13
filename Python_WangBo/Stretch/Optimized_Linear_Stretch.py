import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from tqdm import tqdm


# 像素统计
def Pixel_Statistics(channels, img_paths, bit=8):
    dataset_histogram = np.zeros(shape=(channels, 2 ** bit, 1))
    # pix_max = np.zeros(shape=(channels, 1))
    # pix_min = np.ones(shape=(channels, 1))*2**bit
    # pix_min = 2 ** bit
    # pix_max = 0
    pix_num = 0

    for im_pth in tqdm(img_paths):
        data = gdal.Open(str(im_pth))

        im_width = data.RasterXSize  # 栅格矩阵的列数
        im_height = data.RasterYSize  # 栅格矩阵的行数
        # im_bands = data.RasterCount  # 格栅矩阵的波段
        # im_geotrans = data.GetGeoTransform()  # 仿射变换矩阵
        # im_proj = data.GetProjection()  # 地图的投影信息

        im_data = data.ReadAsArray(xoff=0, yoff=0, xsize=im_width, ysize=im_height)  # 将数据写成数组，对应栅格矩阵 [c, h, w]
        im_data = np.transpose(im_data, axes=(2, 1, 0))  # [w h c]
        # if im_bands > 3:
        #     im_data = im_data[..., :3]

        histogram = np.zeros(shape=(channels, 2 ** bit, 1))

        for ch in range(channels):
            # 若直接使用clacHist()的accumulate=Ture的参数，会导致累计值超出计算机的储存范围
            histogram[ch] = cv2.calcHist([im_data], channels=[ch], mask=None, histSize=[2 ** bit],
                                         ranges=[0, 2 ** bit])
        # pix_max = np.maximum(pix_max, np.max(im_data, axis=2))
        # pix_min = np.minimum(pix_min, np.min(im_data, axis=2))
        # pix_max = np.maximum(pix_max, np.max(im_data))
        # pix_min = np.minimum(pix_min, np.min(im_data))
        dataset_histogram += histogram
        pix_num += im_width * im_height
    dataset_histogram = np.squeeze(dataset_histogram / pix_num)
    # print(dataset_histogram.shape)
    # print(np.sum(dataset_histogram, axis=1))

    # return dataset_histogram, pix_min, pix_max
    return dataset_histogram


# 计算边缘值
def Compute_Edge_Value(channels, dataset_cdf, bit,
                       Min_Percent, Max_Percent,
                       Min_Adjust_Percent, Max_Adjust_Percent):
    # 计算边缘值
    a = np.zeros(shape=(channels, 1, 1))
    b = np.zeros(shape=(channels, 1, 1))
    for ch in range(channels):
        i = 0
        while dataset_cdf[ch, i] < Min_Percent:
            # print(cdf[0, a], Min_Percent)
            i += 1
        a[ch] = i
        # print(a)
        # print(dataset_cdf[ch, i])

        j = 2 ** bit - 1
        while dataset_cdf[ch, j] > Max_Percent:
            # print(cdf[0, b], Max_Percent)
            j -= 1
        b[ch] = j
        # print(b)
        # print(dataset_cdf[ch, j])

    c = a - Min_Adjust_Percent * (b - a)
    d = b + Max_Adjust_Percent * (b - a)
    # print(c.shape, d.shape)
    return a, b, c, d


# 像素统计图
def Plotting_Pixel_Statistics(dataset_name, dataset_histogram, dataset_cdf, in_img_dir, a, b, c, d):
    colors = ['red', 'green', 'blue']
    fig, left_axis = plt.subplots(figsize=(26, 10))
    plt.title(f"{dataset_name}'s Pixel Statistical Histogram", fontsize=20)
    plt.xlabel('pixle value', fontsize=18)
    left_axis.set_ylabel('pixle frequency', fontsize=18)
    right_axis = left_axis.twinx()
    right_axis.set_ylabel('cumulative histogram', fontsize=18)

    dim_1, dim_2 = np.where(dataset_histogram > 0)
    pix_max = np.max(dim_2) + 1
    pix_min = np.min(dim_2)

    xlim_max = int(np.maximum(np.max(dim_2), np.max(d))) + 1
    xlim_min = int(np.minimum(np.min(dim_2), np.min(c))) - 1
    plt.xlim([xlim_min, xlim_max])
    width_val = (1 - 0.1) / len(colors)
    xticks = np.arange(pix_min, pix_max)
    for ch in range(len(colors)):
        left_axis.bar(xticks + width_val * ch, dataset_histogram[ch, pix_min:pix_max], alpha=0.6,
                      width=width_val, color=colors[ch], label=colors[ch])
        right_axis.plot(xticks, dataset_cdf[ch, pix_min:pix_max], color=colors[ch])

        # plt.scatter(a[ch, 0, 0], dataset_cdf[ch, int(a[ch, 0, 0])], c=colors[ch])
        # plt.annotate(f"{int(a[ch, 0, 0])}, {dataset_cdf[ch, int(a[ch, 0, 0])]:.4f}",
        #              (a[ch, 0, 0], dataset_cdf[ch, int(a[ch, 0, 0])]))
        #
        # plt.scatter(b[ch, 0, 0], dataset_cdf[ch, int(b[ch, 0, 0])], c=colors[ch])
        # plt.annotate(f"{int(b[ch, 0, 0])}, {dataset_cdf[ch, int(b[ch, 0, 0])]:.4f}",
        #              (b[ch, 0, 0], dataset_cdf[ch, int(b[ch, 0, 0])]))

        plt.scatter(c[ch, 0, 0], dataset_cdf[ch, int(np.maximum(c[ch, 0, 0], pix_min))],
                    s=20, c=colors[ch])
        # 注释点 edge_left
        plt.annotate(f"({int(c[ch, 0, 0])}, {dataset_cdf[ch, int(np.maximum(c[ch, 0, 0], pix_min))]:.4f})",
                     (c[ch, 0, 0], dataset_cdf[ch, int(np.maximum(c[ch, 0, 0], pix_min))] + 0.02 * (ch - 1)),
                     color=colors[ch])

        plt.scatter(d[ch, 0, 0], dataset_cdf[ch, int(np.minimum(d[ch, 0, 0], pix_max - 1))],
                    s=20, c=colors[ch])
        # 注释点 edge_right
        plt.annotate(f"({int(d[ch, 0, 0])}, {dataset_cdf[ch, int(np.minimum(d[ch, 0, 0], pix_max - 1))]:.4f})",
                     (d[ch, 0, 0], dataset_cdf[ch, int(np.minimum(d[ch, 0, 0], pix_max - 1))] + 0.02 * (ch - 1)),
                     color=colors[ch])

    hist_img_path = os.path.join(in_img_dir.parent, dataset_name + '_distribution.jpg')
    plt.savefig(hist_img_path)


# 优化线性拉伸
def Linear_Stretch(im_data,
                   Edge_l, Edge_r,
                   min_value=0., max_value=1.0):
    # 线性拉伸
    img_correction = np.zeros_like(im_data, dtype=np.float32)
    """[Edge_l, Edge_r] 拉伸到 [min_value, max_value]"""
    index_1 = (im_data < Edge_l)  # (c, w, h)
    img_correction[index_1] = (min_value * index_1)[index_1]
    index_2 = (im_data > Edge_r)
    img_correction[index_2] = (max_value * index_2)[index_2]
    index = (im_data <= Edge_r) & (im_data >= Edge_l)
    img_correction[index] = ((im_data - Edge_l) / (Edge_r - Edge_l) * (max_value - min_value) + min_value)[index]

    img_correction = np.array(img_correction * 255, dtype=np.uint8)
    # print(img_correction.shape, np.min(img_correction), np.max(img_correction))

    return img_correction


def main(in_img_dir=None, in_file_format='tif',
         out_img_dir=None, driver_Format='GTiff', out_file_format='tif',
         bit=8, Channels=4,
         Statistics_Chart=True,
         PreView=False):
    """
    从这些数据中计算出一个相对累积的直方图。
    查找与相对累积直方图中的“Min_Percent”和“Max_Percent”对应的数据值,并分别将它们标记为a和b。
    大多数像素的数据值都在a到b的中音范围内。
    通过减少a的"Min_Adjust_Percent"来计算黑点c
    c = a - Min_Adjust_Percent * (b - a)
    通过增加b的"Max_Adjust_Percent"来计算白点d
    d = b + Max_Adjust_Percent * (b - a)

    Parameters
    ----------
    in_img_dir : str
        待拉伸图像的文件夹位置; 例如:r"F:\Optimized_Linear_Stretch\dataset_test"
    in_file_format : str
        待拉伸图像的文件名的后缀; 例如："tif" "img" "JP2"

    bit : int
        影像的位,8bit, 16bit
    Channels: int
        需要拉伸的通道数量
    Statistics_Chart: bool
        True: 绘出该影像文件夹的像素统计图，并保存在该影像文件夹的所在的目录下
        False: 不保存像素统计图

    PreView: bool
        True: 即为预览模式, 输出不落地
        False: 批处理模式， 将拉伸后的影像保存在out_img_dir

    out_img_dir : str
        图像拉伸变换后保存位置; 例如:r"F:\Optimized_Linear_Stretch\dataset_test_out"
    driver_Format : str
        影像驱动,不同格式数据文件需要用对应的影像驱动driver来生成; 例如：.tif格式对应的driver是GTiff，.img格式对应的driver是HFA
        当值为None时,自适应匹配原图的driver
    out_file_format : str
        图像拉伸变换后保存时的文件名后缀,需要与“driver_Format” 对应配套使用
        当driver_Format值为None时,自适应匹配原图的后缀名

    """

    dataset_name = in_img_dir.split('\\')[-1]
    in_img_dir = Path(in_img_dir)
    img_paths = list(in_img_dir.glob('*.' + in_file_format))  # 要求文件夹内的格式都是一样

    histogram_pth = os.path.join(in_img_dir.parent, dataset_name + '_histogram.npy')
    if os.path.exists(histogram_pth):
        print(f'加载像素分布表：{dataset_name}_histogram.npy ...')
        dataset_histogram = np.load(histogram_pth)
    else:
        print(f'统计{dataset_name}的像素分布...')
        dataset_histogram = Pixel_Statistics(Channels, img_paths, bit)
        np.save(histogram_pth, dataset_histogram)

    dataset_cdf = np.zeros_like(dataset_histogram)
    for x in range(2 ** bit):
        """累计分布;F(x) = P(X <= x)"""
        dataset_cdf[:, x:x + 1] = np.sum(dataset_histogram[:, :x + 1], axis=1, keepdims=True)

    print("Please enter the parameters and debug:")
    print("Min_Percent: float, range(0, 1), example: 0.025")
    Min_Percent = np.float32(input())
    print("Max_Percent: float, range(0, 1), example: 0.99")
    Max_Percent = np.float32(input())
    # print("min_value: range(0, 1):")
    # min_value = np.float32(input())
    # print("max_value: range(0, 1):")
    # max_value = np.float32(input())
    print("Min_Adjust_Percent: float, range(0, 1), example: 0.1")
    Min_Adjust_Percent = np.float32(input())
    print("Max_Adjust_Percent: float, range(0, 1), example: 0.5")
    Max_Adjust_Percent = np.float32(input())

    # 参数调试：
    if PreView:  # 预览模式

        a, b, c, d = Compute_Edge_Value(Channels, dataset_cdf, bit,
                                        Min_Percent, Max_Percent,
                                        Min_Adjust_Percent, Max_Adjust_Percent)
        print(f"edge_left:\n{c},\nedge_right:\n{d}")
        if Statistics_Chart:
            print("绘制RGB像素频率柱状图...")

            Plotting_Pixel_Statistics(dataset_name, dataset_histogram, dataset_cdf, in_img_dir, a, b, c, d)

        img_paths = random.sample(img_paths, len(img_paths))
        for i, im_pth in enumerate(img_paths):
            # 逐张影像拉伸
            data = gdal.Open(str(im_pth))
            im_width = data.RasterXSize  # 栅格矩阵的列数
            im_height = data.RasterYSize  # 栅格矩阵的行数

            im_data = data.ReadAsArray(xoff=0, yoff=0, xsize=im_width, ysize=im_height)  # 将数据写成数组，对应栅格矩阵 [c, h, w]

            img_correction = Linear_Stretch(im_data[:3, ...], c[:3, ...], d[:3, ...])  # [c, h, w]

            # 预览只展示RGB三通道成像
            img_preview = np.transpose(img_correction[::-1, ...], axes=(1, 2, 0))  #
            cv2.namedWindow(f'Preview:{i}', cv2.WINDOW_FREERATIO)

            if (im_width > 2400) or (im_height > 2400):
                im_width = im_width / 10
                im_height = im_height / 10
            cv2.resizeWindow(f'Preview:{i}', width=int(im_width), height=int(im_height))
            cv2.imshow(f'Preview:{i}', img_preview)
            cv2.waitKey()
            cv2.destroyWindow(f'Preview:{i}')

    else:
        out_img_dir = Path(out_img_dir)
        out_img_dir.mkdir(exist_ok=True)
        a, b, c, d = Compute_Edge_Value(Channels, dataset_cdf, bit,
                                        Min_Percent, Max_Percent,
                                        Min_Adjust_Percent, Max_Adjust_Percent)
        for im_pth in tqdm(img_paths):
            # 逐张影像拉伸
            data = gdal.Open(str(im_pth))
            im_width = data.RasterXSize  # 栅格矩阵的列数
            im_height = data.RasterYSize  # 栅格矩阵的行数
            im_bands = data.RasterCount  # 格栅矩阵的波段
            im_data = data.ReadAsArray(xoff=0, yoff=0, xsize=im_width, ysize=im_height)  # 将数据写成数组，对应栅格矩阵 [c, h, w]

            img_correction = Linear_Stretch(im_data[:Channels, ...],
                                            c, d, min_value=0.01, max_value=0.99)

            # 写入数据
            im_geotrans = data.GetGeoTransform()  # 仿射变换矩阵
            im_proj = data.GetProjection()  # 地图的投影信息

            # 数据格式(.tif、.img)后缀 需要用对应的驱动driver(GTiff、HFA)来生成
            if driver_Format:
                im_driver = gdal.GetDriverByName(driver_Format)  # 参数给需要转换的目标格式
            else:
                im_driver = data.GetDriver()
                out_file_format = in_file_format

            image_name = im_pth.name
            image_name = image_name.split('.')[:-1]
            image_name.append(out_file_format)
            image_name = '.'.join(image_name)
            out_im_pth = out_img_dir / image_name

            # if 'int8' in im_data.dtype.name:
            #     im_dtype = gdal.GDT_Byte
            # elif 'int16' in im_data.dtype.name:
            #     im_dtype = gdal.GDT_UInt16
            # else:
            #     im_dtype = gdal.GDT_Float32

            out_data = im_driver.Create(str(out_im_pth), im_width, im_height, im_bands, gdal.GDT_Byte)

            out_data.SetGeoTransform(im_geotrans)  # 写入放射变换
            out_data.SetProjection(im_proj)  # 写入投影

            # 写入数组数据
            for band in range(Channels):
                out_data.GetRasterBand(band + 1).WriteArray(img_correction[band])

            del out_data

    print('all done')


if __name__ == '__main__':
    """8bit 3channel"""
    # in_image_dir = r'F:\data_analysis\Optimized_Linear_Stretch\08bit3channel'
    # out_image_dir = r'F:\data_analysis\Optimized_Linear_Stretch\08bit3channel_out'
    in_image_dir = r'F:\data_analysis\Band_Fusion'
    out_image_dir = r'F:\data_analysis\Band_Fusion_out'

    main(in_img_dir=in_image_dir, in_file_format='tif',
         out_img_dir=out_image_dir, driver_Format='GTiff', out_file_format='tif',
         bit=8, Channels=3,
         Statistics_Chart=True,

         PreView=False)

    """8bit 4channel"""
    # in_image_dir = r'F:\data_analysis\Optimized_Linear_Stretch\08bit4channel'
    # out_image_dir = r'F:\data_analysis\Optimized_Linear_Stretch\08bit4channel_out'
    # main(in_img_dir=in_image_dir, in_file_format='tif',
    #      out_img_dir=out_image_dir, driver_Format='GTiff', out_file_format='tif',
    #      bit=8, Channels=4,
    #      Statistics_Chart=True,
    #
    #      PreView=True)

    """16bit 4channel"""
    # in_image_dir = r'F:\data_analysis\Optimized_Linear_Stretch\16bit4channel'
    # out_image_dir = r'F:\data_analysis\Optimized_Linear_Stretch\16bit4channel_out'

    # in_image_dir =r"F:\data_analysis\Band_Fusion"
    # out_image_dir =r"F:\data_analysis\Band_Fusion_out2"
    #
    # main(in_img_dir=in_image_dir, in_file_format='tif',
    #      out_img_dir=out_image_dir, driver_Format='GTiff', out_file_format='tif',
    #      bit=16, Channels=1,
    #      Statistics_Chart=True,
    #
    #      PreView=True)

    """Batch_test"""
    # in_image_dir = r'F:\data_analysis\Optimized_Linear_Stretch\Batch_test'
    # out_image_dir = r'F:\data_analysis\Optimized_Linear_Stretch\Batch_test_out'
    # main(in_img_dir=in_image_dir, in_file_format='tif',
    #      out_img_dir=out_image_dir, driver_Format='GTiff', out_file_format='tif',
    #      bit=8, Channels=3,
    #      Statistics_Chart=True,
    #
    #      PreView=True)

    # pth = r"F:\data_analysis\Band_Fusion\single_band_HR.tif"
    # # mul_data = cv2.imread(pth, -1)
    #
    # import tifffile.tifffile as tiff
    # mul_data = tiff.imread(pth)
    # print(mul_data.shape)
    # # mul_data = mul_data[..., :3]
    # # mul_data = mul_data[..., ::-1]
    # mul_data = (mul_data-np.min(mul_data))/(np.max(mul_data)-np.min(mul_data))
    # mul_data = np.reshape(mul_data, (6999, 7379, 1))
    # mul_data = np.concatenate((mul_data, mul_data, mul_data), axis=2)
    # print(mul_data.shape)
    # cv2.imshow('mul', np.uint8(mul_data*255))
    # cv2.waitKey()