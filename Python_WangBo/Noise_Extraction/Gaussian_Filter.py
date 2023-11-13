import cv2
import numpy as np


img_path = r"F:\data_analysis\illumination_contrast_balancing\LW_out\test4.png"
img_data = cv2.imread(img_path)/255
h, w, c = img_data.shape
cv2.imshow('img_data', img_data)

img_path2= r"F:\data_analysis\illumination_contrast_balancing\LW_out\test2.png"
img_data2=cv2.imread(img_path2)/255

new_data = np.zeros_like(img_data2)
for k in range(c):
    img_band = img_data[..., k]

    n = 3
    img_band_denoise = cv2.blur(img_band, ksize=(n, n))
    # print(np.std(img_data))
    # img_data_denoise = cv2.GaussianBlur(img_data,sigmaX=np.std(img_data), sigmaY=np.std(img_data), ksize=(n,n))
    # img_band_denoise = cv2.medianBlur(img_band, ksize=n)
    cv2.imshow(f'img_band_denoise {k}', img_band_denoise )

    band_noise = img_band - img_band_denoise
    print(np.max(band_noise), np.min(band_noise))
    cv2.imshow(f'band_noise {k}', band_noise)

    img_band2 = img_data2[..., k]
    new_data[..., k] = img_band2 + band_noise
    print(np.max(new_data[..., k]), np.min(new_data[..., k]))
    cv2.imshow(f'new_data[{k}]', new_data[..., k])

cv2.imshow('new_data', new_data)
cv2.waitKey()