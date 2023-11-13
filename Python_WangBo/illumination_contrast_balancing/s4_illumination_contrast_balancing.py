import time

import cv2
import numpy as np
from scipy import signal

epsilon = 1e-8


def Gauss_LowPass_Filter(u, v, h, w, D_0):
    return np.exp(-1 / 2 * ((u - h / 2)**2 + (v - w / 2)**2) / (D_0**2))


def Definition_value(img_ref_block):
    """
    计算 Definition值:文中解释卫清晰度
    """
    M, N, _ = img_ref_block.shape
    delta_x = img_ref_block[1:, ...] - img_ref_block[:-1, ...]  # [M-1, N, _]
    delta_y = img_ref_block[:, 1:, ...] - img_ref_block[:, :-1,
                                                        ...]  # [M, N-1, _]

    return np.sum(np.sqrt((delta_x[:, 1:N]**2 + delta_y[1:M, :]**2) / 2),
                  axis=(0, 1, 2)) / ((M - 1) * (N - 1) + epsilon)


def illumination_contrast_balancing(img_data,
                                    D0=4,
                                    offset=0,
                                    blk_size=21,
                                    t=0.005,
                                    gamma=0.05,
                                    pix_Range=(30, 225)):
    """
    针对卫星图像的图像平衡方法。
    该方法调整每个像素处邻域的均值和标准差，包括消除粗光背景、图像平衡和最大平均最小辐射校正三个步骤。

    step1. 在频域内大致消除了光的背景。
    step2. 利用两个平衡因子和线性变换，自适应地调整每个像素的局部均值和标准差。
    step3. 经过最大平均最小辐射校正，利用保色因子得到平衡图像。

    Parameters
    ----------
    img_data: 输入数据,ndarry格式
    D0: 高斯低频滤波的距离,D0∈[0, min(h, w)],值越大，过滤的低频（背景）信息越多
    offset: 高度偏移量,修正剔除粗光背景后整体亮度,offset∈[-128, 128],值越大，整体越亮
    blk_size: 调整均值、方差的领域值,blk_size∈[0, min(h, w)],值越大尺寸越大，计算量越大，图像越平滑
    t: 尺度参数,t∈[0, 1], 值越大，保留极亮、极暗的像素点越少
    gamma: 提高图像的颜色效果,lgamma∈[0, 1],值越大,越多的像素值超过255
    
    默认参数都是论文给的实验数据
    """

    h, w, c = img_data.shape
    '''
    step1: Coarse_Light Background Elimination
    '''
    """构造高斯低通滤波器"""
    GLpFilter_h = np.reshape(np.arange(0, h), newshape=(1, h))
    GLpFilter_w = np.reshape(np.arange(0, w), newshape=(1, w))
    GLpFilter_idx = np.array(np.meshgrid(GLpFilter_h, GLpFilter_w))
    GLpFilter_idx = np.transpose(GLpFilter_idx, axes=(2, 1, 0))  # [h, w, 2]
    GLpFilter = Gauss_LowPass_Filter(GLpFilter_idx[..., 0],
                                     GLpFilter_idx[..., 1], h, w, D0)

    img_data_elim = np.zeros_like(img_data, dtype='float32')

    for k in range(c):
        img_band = img_data[..., k]
        img_band_dft = np.fft.fft2(
            img_band)  # 傅里叶 空域-->频域 [512 512] dtype:complex
        img_band_dft_shift = np.fft.fftshift(img_band_dft)  # 低频中移动
        """查看频谱图"""
        result = 20 * np.log(np.abs(img_band_dft_shift))
        cv2.imshow('fft', (result - np.min(result)) /
                   (np.max(result) - np.min(result)))
        """滤波"""
        B_light = np.fft.ifft2(np.fft.ifftshift(
            img_band_dft_shift * GLpFilter))  # [h, w] complex128
        """去粗光背景"""
        new = np.zeros(shape=(h, w))
        for i in range(h):
            for j in range(w):
                real = B_light[i, j].real
                imag = B_light[i, j].imag
                new[i, j] = np.sqrt(real**2 + imag**2)

        img_data_elim[..., k] = img_band - new + np.mean(
            img_band) + offset  # [476, 476] float64
    cv2.imshow('img_elimination', img_data_elim)
    cv2.moveWindow('img_elimination', (10 + w) * 1, 10)
    # cv2.waitKey()
    """
    step2: image balanceing
    """

    delta_x = img_data_elim[1:, :, :] - img_data_elim[:-1, :, :]  # [h-1, w, _]
    delta_y = img_data_elim[:, 1:, :] - img_data_elim[:, :-1, :]  # [h, w-1, _]
    residual = np.sqrt(
        (delta_x[:, 1:w]**2 + delta_y[1:h, :]**2) / 2)  # [h-1, w-1, _]
    '''scipy 2d卷积'''
    def_filter = np.ones(shape=(blk_size - 1, blk_size - 1),
                         dtype=np.float32) / ((blk_size - 1)**2)
    defination_value = np.zeros(shape=((h - 1) - (blk_size - 1) + 1,
                                       (w - 1) - (blk_size - 1) + 1),
                                dtype=np.float32)

    nbr_filter = np.ones(shape=(blk_size, blk_size),
                         dtype=np.float32) / (blk_size**2)
    Mean_nbr = np.zeros_like(img_data, dtype=np.float32)
    Std_nbr = np.zeros_like(img_data, dtype=np.float32)

    for ci in range(c):
        defination_value += signal.convolve2d(residual[..., ci],
                                              def_filter,
                                              mode='valid')
        """逐像素计算 以blk_size为邻域的 Mean & Std"""
        Mean_nbr[..., ci] = signal.convolve2d(img_data_elim[..., ci],
                                              nbr_filter,
                                              mode='same',
                                              boundary='symm')
        Std_nbr[..., ci] = np.sqrt(
            signal.convolve2d((img_data_elim[..., ci] - Mean_nbr[..., ci])**2,
                              nbr_filter,
                              mode='same',
                              boundary='symm'))

    ref_index = np.unravel_index(
        np.argmax(defination_value),
        defination_value.shape)  # 根据最大De值 找到指定ref_Block

    # todo 应该改以ref_index为中心店点 找长宽为blk_size尺寸的ref_block
    # ref_block = img_data_elim[ref_index[0] - int(blk_size / 2):ref_index[0] + int(blk_size / 2),
    #                           ref_index[1] - int(blk_size / 2):ref_index[1] + int(blk_size / 2)]
    ref_block = img_data_elim[ref_index[0]:ref_index[0] + blk_size,
                              ref_index[1]:ref_index[1] + blk_size]

    cv2.imshow("ref_block", ref_block)
    cv2.moveWindow("ref_block", (10 + w) * 1, 50 + h)
    cv2.waitKey()

    Mean_ref = np.mean(ref_block, axis=(0, 1))
    Std_ref = np.std(ref_block, axis=(0, 1))
    """计算 weights"""
    w_s = Std_ref / (Std_ref + Std_nbr)
    w_m = Mean_ref / (Mean_ref + Mean_nbr)
    """balancing factors α β"""
    alpha = w_s * Std_ref / (w_s * Std_nbr + (1 - w_s) * Std_ref)
    beta = w_m * Mean_ref + (1 - w_m - alpha) * Mean_nbr

    img_data_balancing = alpha * img_data_elim + beta
    del img_data_elim
    cv2.imshow("img_balancing", img_data_balancing)
    cv2.moveWindow('img_balancing', (10 + w) * 2, 10)
    # cv2.waitKey()
    """
    step3: Max-Mean-Min Radiation Correction
    """
    img_data_correction = np.zeros_like(img_data, dtype=np.float32)
    T = t * h * w
    """根据阈值T, 找到较亮的、较暗的 部分像素点集的边界值max_ min_"""
    img_gray = cv2.cvtColor(np.uint8(img_data_balancing * 255),
                            cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([img_gray],
                             channels=[0],
                             mask=None,
                             histSize=[256],
                             ranges=[0, 256])
    histogram = np.squeeze(histogram)  # [c, 256]
    i = 0
    cdf_i = 0
    while cdf_i <= T:
        i += 1
        cdf_i = np.sum(histogram[:i])
    min_ = (i - 1) / 255

    j = 256
    cdf_j = 0
    while cdf_j <= T:
        j -= 1
        cdf_j = np.sum(histogram[j:256])
    max_ = (j + 1) / 255

    meanv = np.mean(img_data_balancing, axis=(0, 1))
    """linear transform：分段线性函数"""
    pix_Range = pix_Range/255
    img_data_correction[np.where(img_data_balancing < min_)] = pix_Range[0]
    img_data_correction[np.where(img_data_balancing >= max_)] = pix_Range[1]

    index_1 = (min_ <= img_data_balancing) & (img_data_balancing < meanv
                                              )  # [h, w, 3]
    img_data_correction[index_1] = (pix_Range[0] * index_1 +
                                    (img_data_balancing * index_1 - min_) *
                                    (((meanv - pix_Range[0]) /
                                      (meanv - min_)) * index_1))[index_1]
    print(
        f'max:{np.max(img_data_correction, axis=(0, 1))}, min:{np.min(img_data_correction, axis=(0, 1))}'
    )

    index_2 = (meanv <= img_data_balancing) & (img_data_balancing < max_)
    img_data_correction[index_2] = (meanv * index_2 +
                                    ((img_data_balancing - meanv) * index_2) *
                                    ((pix_Range[1] - meanv) /
                                     (max_ - meanv)) * index_2)[index_2]

    print(
        f'max:{np.max(img_data_correction, axis=(0, 1))}, min:{np.min(img_data_correction, axis=(0, 1))}'
    )
    """"""
    maxv = np.argmax(histogram) / 255

    if maxv < (128 / 255):
        lambda_ = (maxv / (128 / 255))**gamma
    else:
        lambda_ = ((128 / 255) / maxv)**gamma
    print(maxv, lambda_)
    img_data_correction = np.minimum((img_data_correction**lambda_), 1.)
    print(
        f'max:{np.max(img_data_correction, axis=(0, 1))}, min:{np.min(img_data_correction, axis=(0, 1))}'
    )
    del img_data_balancing

    # return np.uint8(img_data_correction)
    return img_data_correction


if __name__ == '__main__':
    import os

    """文件夹 批处理"""
    # img_dir = r'F:\data_analysis\illumination_contrast_balancing\max_mean_min'  # 论文所给影像
    # save_dir = r'F:\data_analysis\illumination_contrast_balancing\Correction'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # for name in os.listdir(img_dir):
    #     img_pth = os.path.join(img_dir, name)
    #     image_data = cv2.imread(img_pth, -1) / 255

    #     cv2.imshow('img_o', image_data)
    #     cv2.moveWindow('img_o', 10, 10)

    #     t1 = time.time()
    #     image_correct = illumination_contrast_balancing(
    #         image_data[..., :3],
    #         D0=2,
    #         offset=0,
    #         blk_size=31,
    #         t=0.005,
    #         gamma=0.2,
    #         pix_Range=[10, 245])
    #     t2 = time.time()
    #     print(
    #         f'尺寸: {image_data.shape}, 耗时：{t2 - t1}, max:{np.max(image_correct, axis=(0, 1))}, '
    #         f'min:{np.min(image_correct, axis=(0, 1))}')

    #     save_pth = os.path.join(save_dir, name)
    #     cv2.imwrite(save_pth, np.uint8(image_correct * 255))

    #     cv2.imshow('img_correct', image_correct)
    #     cv2.moveWindow('img_correct', (10 + image_data.shape[1]) * 3, 10)
    #     cv2.waitKey()

    """单张img"""
    img_pth = r"F:\data_analysis\illumination_contrast_balancing\max_mean_min\test1.png"
    # out_pth = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\img_balance_to_target\07_08-4_target.tif'

    image_data = cv2.imread(img_pth, -1)
    h, w, c = image_data.shape
    # image_data = cv2.resize(image_data, (int(w/20), int(h/20)))

    cv2.imshow('img_o', image_data)
    cv2.waitKey()

    t1 = time.time()
    image_correct = illumination_contrast_balancing(
        image_data[..., :3] / 255,
        D0=8,
        offset=0,
        blk_size=31,
        t=0.005,
        gamma=0.02,
        pix_Range=[20, 235])
    t2 = time.time()
    cv2.imshow('image_correct', image_correct)
    cv2.waitKey()
    print(
        f'尺寸: {image_data.shape}, 耗时：{t2 - t1}, max:{np.max(image_correct, axis=(0, 1))}, '
        f'min:{np.min(image_correct, axis=(0, 1))}')
    # cv2.imwrite(out_pth, np.uint8(image_correct*255))
