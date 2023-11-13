import cv2
import numpy as np


def np_scale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


def Gaussian_HighPass_filter(u, v, h, w, D_0):
    return 1 - np.exp(-1 / 2 * ((u - h / 2) ** 2 + (v - w / 2) ** 2) / (D_0 ** 2))

def Gaussian_LowPass_filter(u, v, h, w, D_0):
    return np.exp(-1 / 2 * ((u - h / 2) ** 2 + (v - w / 2) ** 2) / (D_0 ** 2))


img_path = r"F:\data_analysis\illumination_contrast_balancing\LW_out\test4.png"
img_data = cv2.imread(img_path)/255

img_path2= r"F:\data_analysis\illumination_contrast_balancing\LW_out\test2.png"
img_data2=cv2.imread(img_path2)/255

cv2.imshow('img', img_data)

h, w, c = img_data.shape

"""构造高斯高通通滤波器"""
GLpFilter_h = np.reshape(np.arange(0, h), newshape=(1, h))
GLpFilter_w = np.reshape(np.arange(0, w), newshape=(1, w))
GLpFilter_idx = np.array(np.meshgrid(GLpFilter_h, GLpFilter_w))
GLpFilter_idx = np.transpose(GLpFilter_idx, axes=(2, 1, 0))  # [h, w, 2]
# GHpFilter = Gaussian_HighPass_filter(GLpFilter_idx[..., 0], GLpFilter_idx[..., 1], h, w, D_0=16)
# print(GHpFilter.shape, np.max(GHpFilter), np.min(GHpFilter))  # [h, w] 值域[0, 1]
# cv2.imshow('GHpF', GHpFilter)
GLpFilter = Gaussian_LowPass_filter(GLpFilter_idx[..., 0], GLpFilter_idx[..., 1], h, w, D_0=4)
print(GLpFilter.shape, np.max(GLpFilter), np.min(GLpFilter))  # [h, w] 值域[0, 1]
cv2.imshow('GHpF', GLpFilter)


noise = np.zeros_like(img_data, dtype='float32')
new_data = np.zeros_like(img_data2, dtype='float32')
for k in range(c):
    img_band = img_data[..., k]
    img_band2 = img_data2[..., k]
    print('img_band:', np.min(img_band), np.max(img_band))
    img_band_dft = np.fft.fft2(img_band)  # 傅里叶 空域-->频域 [512 512]
    img_band2_dft =np.fft.fft2(img_band2)
    print('傅里叶变换频域的结果：', img_band_dft.shape)
    img_band_dft_shift = np.fft.fftshift(img_band_dft)  # 低频中移动
    print('傅里叶变换频域低频中移的结果：', img_band_dft_shift.shape)

    # """查看频谱图"""
    # # result = 20 * np.log(cv2.magnitude(img_data_dft_shift[:, :, 0], img_data_dft_shift[:, :, 1]))
    # result = 20 * np.log(cv2.magnitude(img_band_dft[:, :, 0], img_band_dft[:, :, 1]))
    # result2 = 20 * np.log(cv2.magnitude(img_band2_dft[:, :, 0], img_band2_dft[:, :, 1]))
    # print(np.max(result), np.min(result), result.shape)
    # cv2.imshow('fft', (result-np.min(result))/(np.max(result)-np.min(result)))
    # cv2.imshow('fft2', (result2-np.min(result2))/(np.max(result2)-np.min(result2)))

    """从频域滤波 得到高频频谱-->频谱还原-->逆傅里叶变换 得到空域"""
    # B_light = np.fft.ifft2(np.fft.ifftshift(img_band_dft_shift * GHpFilter))  # [h, w], complex128

    Band_light = np.fft.ifft2(img_band_dft * GLpFilter)  # [h, w], complex128
    Band2 = np.fft.ifft2(img_band2_dft + img_band_dft * GLpFilter)  # img_data1的噪声注入到img_data2中

    noise_band = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            real = Band_light[i, j].real  # 取实部
            # imag = B_light[i, j].imag  # 虚部信息基本忽略
            noise_band[i, j] = real

            new_data[i, j, k] = Band2[i,j].real
            

    print('B_light的实部：', noise_band.shape, noise_band.dtype, np.max(noise_band), np.min(noise_band))
    noise[..., k] = noise_band

cv2.imshow('noise', np_scale(noise))

"""原图+噪声"""
cv2.imshow('img2', img_data2)
print(f'img2: mean={np.mean(img_data2)}, std={np.std(img_data2)}') 

cv2.imshow('img2+noise_v1', (img_data2 + noise))
print(f'img2+noise_v1: mean={np.mean(img_data2 + noise)}, std={np.std(img_data2 + noise)}') 

cv2.imshow('img2+noise2', new_data)
print(f'img2+noise_v2: mean={np.mean(new_data)}, std={np.std(new_data)}') 

cv2.waitKey()


