# from numpy import fft as npfft
import cv2
import numpy as np

# img_file = r'F:\data_analysis\illumination_contrast_balancing_shrinkly'
# img_data_list = []
# for i, img_name in enumerate(os.listdir(img_file)):
#     print(i, img_name)
#     img_path = os.path.join(img_file, img_name)
#     img_data_i = cv2.imread(img_path, -1)
#     cv2.imshow(f'img_data_{i}', img_data_i)
#     cv2.waitKey()
#     img_data_list.append(img_data_i)
#
# print(img_data_list)
#
# img_data = np.zeros(shape=(512, 512, 3), dtype='uint8')
#
# img_data[:256 * 1, :256 * 1, ...] = img_data_list[0]
# img_data[:256 * 1, 256 * 1: 256 * 2, ...] = img_data_list[1]
# img_data[256 * 1: 256 * 2, :256 * 1, ...] = img_data_list[2]
# img_data[256 * 1: 256 * 2, 256 * 1: 256 * 2, ...] = img_data_list[3]
#
# cv2.imshow('img_data', img_data)
# cv2.waitKey()
# save_name = r'61011967_3band_shrank.tif'
# cv2.imwrite(os.path.join(img_file, save_name), img_data)

# img_path = r'F:\data_analysis\illumination_contrast_balancing\band_3.png'  #  61011967_3band_shrank.tif
# img_path = r'F:\data_analysis\illumination_contrast_balancing\test1_o.png'
img_path = r"F:\data_analysis\illumination_contrast_balancing\max_mean_min\test1.png"
img_data = cv2.imread(img_path)  # [512, 512, 3]
print(img_data.shape)

h, w, c = img_data.shape
# img_data_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img_data)
img_data_elim = np.zeros_like(img_data, dtype='float32')
img_data_elim_NoShift = np.zeros_like(img_data, dtype='float32')
img_background = np.zeros_like(img_data, dtype='float32')
img_background_NoShift = np.zeros_like(img_data, dtype='float32')


def gaussian_lowpass_filter(u, v, h, w, D_0):
    return np.exp(-1 / 2 * ((u - h / 2) ** 2 + (v - w / 2) ** 2) / (D_0 ** 2))


def gaussian_lowpass_filter_Noshift(u, v, h, w, D0):
    d = (np.abs(v-h/2)-(h-1)/2)**2 + (np.abs(u-w/2)-(w-1)/2)**2
    return np.exp(-1/2 * d/D0**2)


"""构造高斯低通滤波器"""
GLpFilter_h = np.reshape(np.arange(0, h), newshape=(1, h))
GLpFilter_w = np.reshape(np.arange(0, w), newshape=(1, w))
GLpFilter_idx = np.array(np.meshgrid(GLpFilter_h, GLpFilter_w))
GLpFilter_idx = np.transpose(GLpFilter_idx, axes=(2, 1, 0))  # [h, w, 2]
print('高斯低通滤波:', GLpFilter_idx.shape)

GLpFilter = gaussian_lowpass_filter(GLpFilter_idx[..., 0], GLpFilter_idx[..., 1], h, w, D_0=4)
GLpFilter_Noshift = gaussian_lowpass_filter_Noshift(GLpFilter_idx[..., 0], GLpFilter_idx[..., 1], h, w, D0=4)
print(GLpFilter.shape, np.max(GLpFilter), np.min(GLpFilter))
print(GLpFilter_Noshift.shape, np.max(GLpFilter_Noshift), np.min(GLpFilter_Noshift))

cv2.imshow('GLpF', GLpFilter)
cv2.imshow('GLpF_NoShift', GLpFilter_Noshift)

for k in range(c):
    img_band = img_data[..., k]
    # img_band_dft = cv2.dft(np.float32(img_band), flags=cv2.DFT_COMPLEX_OUTPUT)
    # img_data_dft = fft.fft2(np.float32(img_data_gray))  # [512 512]
    img_band_dft = np.fft.fft2(img_band)  # 傅里叶 空域-->频域 [512 512]
    print('傅里叶变换频域的结果：', img_band_dft.shape)
    img_band_dft_shift = np.fft.fftshift(img_band_dft)  # 低频中移动

    """查看频谱图"""

    # # result = 20 * np.log(np.sqrt((img_band_dft_shift.real ** 2) + (img_band_dft_shift.imag ** 2)))
    # result = 20 * np.log(np.abs(img_band_dft_shift))
    # cv2.imshow('fft', (result-np.min(result))/(np.max(result)-np.min(result)))

    """滤波"""
    B_light = np.fft.ifft2(np.fft.ifftshift(img_band_dft_shift * GLpFilter))
    B_light_NoShift = np.fft.ifft2(img_band_dft * GLpFilter_Noshift)
    print('B_light:', B_light.shape, B_light.dtype)
    print('B_light_NoShift:', B_light_NoShift.shape, B_light_NoShift.dtype)

    new = np.zeros(shape=(h, w))
    new_NoShift = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            real = B_light[i, j].real
            imag = B_light[i, j].imag
            new[i, j] = np.sqrt(real**2 + imag**2)

            real_NoShift = B_light_NoShift[i, j].real
            imag_NoShift = B_light_NoShift[i, j].imag
            new_NoShift[i, j] = np.sqrt(real_NoShift ** 2 + imag_NoShift ** 2)

    print('B_light的实部：', new.shape, new.dtype, np.max(new), np.min(new))
    print('B_light_NoShift的实部：', new_NoShift.shape, new_NoShift.dtype, np.max(new_NoShift), np.min(new_NoShift))

    img_background[..., k] = (new - np.min(new)) / (np.max(new) - np.min(new))
    img_background_NoShift[..., k] = (new_NoShift - np.min(new_NoShift)) / (np.max(new_NoShift) - np.min(new_NoShift))

    img_band_elim = img_band - new + np.mean(img_band)  # mean: 73.13
    img_band_elim_NoShift = img_band - new_NoShift + np.mean(img_band)  # mean: 73.13

    print('x_band\'', img_band_elim.shape, img_band_elim.dtype, np.max(img_band_elim), np.min(img_band_elim))
    img_data_elim[..., k] = (img_band_elim - np.min(img_band_elim)) / (np.max(img_band_elim) - np.min(img_band_elim))
    img_data_elim_NoShift[..., k] = (img_band_elim_NoShift - np.min(img_band_elim_NoShift)) / (np.max(img_band_elim_NoShift) - np.min(img_band_elim_NoShift))


print('x\'', img_data_elim.shape, img_data_elim.dtype, np.max(img_data_elim), np.min(img_data_elim))
cv2.imshow('elim', img_data_elim)
cv2.imshow('elim_NoShift', img_data_elim_NoShift)
# save_path = r'F:\data_analysis\illumination_contrast_balancing_shrinkly\band_3_elim.png'
# save_path = r'F:\data_analysis\illumination_contrast_balancing\test4_elim.tif'
# cv2.imwrite(save_path, np.uint8(img_data_elim*255))
cv2.imshow('background', img_background)
cv2.imshow('background_NoShift', img_background_NoShift)
cv2.waitKey()


