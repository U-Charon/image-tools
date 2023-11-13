
import cv2
import numpy as np

# img_path = r'F:\data_analysis\illumination_contrast_balancing\band_3_balancing.png'
img_path = r'F:\data_analysis\illumination_contrast_balancing\test4_balancing.tif'
img_data = cv2.imread(img_path, -1)
cv2.imshow('balancing', img_data)
cv2.waitKey()

h, w, c = img_data.shape
t = 0.0001
gamma = 0.005
T = t * h * w
img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
print('img_gray', np.min(img_gray), np.max(img_gray))
# histogram = np.zeros(shape=(1, 256, 1))
# 若直接使用clacHist()的accumulate=Ture的参数，会导致累计值超出计算机的储存范围
histogram = cv2.calcHist([img_gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256]) / (h * w)
print(histogram.shape)  # [256, 1]

histogram = np.squeeze(histogram)  # [c, 256]
print(histogram.shape, np.sum(histogram))  # [256,]

i = 0
cdf_i = 0
while cdf_i <= T:
    i += 1
    cdf_i = np.sum(histogram[:i])
    # print('求min', cdf_i, i)
min_ = i-1
print(min_)

img_data[np.where(img_data < min_)] = 0
print('0000', img_data.shape, np.max(img_data), np.min(img_data))
cv2.imshow('a', img_data)


j = 256
cdf_j = 0
while cdf_j <= T:
    j -= 1
    cdf_j = np.sum(histogram[j:256])
    # print('求max', cdf_j, j)
max_ = j+1
print(max_)

img_data[np.where(img_data >= max_)] = 255
print('1111', img_data.shape, np.max(img_data), np.min(img_data))
cv2.imshow('b', img_data)


meanv = np.mean(img_data, axis=(0, 1))
print(meanv)

index = (min_ <= img_data) & (img_data < meanv)  # [h, w, 3]
print(index.shape, (img_data*index).shape, ((meanv-min_)*index).shape, )
img_data[index] = ((img_data*index - min_)*((meanv/(meanv-min_))*index))[index]
print('2222', img_data.shape, np.max(img_data), np.min(img_data))
cv2.imshow('c', np.uint8(img_data))

index = (meanv <= img_data) & (img_data < max_)
print(index.shape, (img_data * index).shape, meanv.shape, )
img_data[index] = (meanv*index + ((img_data-meanv)*index)*((255-meanv)/(max_-meanv))*index)[index]
print('3333', img_data.shape, np.max(img_data), np.min(img_data))
cv2.imshow('d', np.uint8(img_data))


maxv = np.argmax(histogram)
print('maxv:', maxv)

if maxv < 128:
    lambda_ = (maxv/128)**gamma
else:
    lambda_ = (128/maxv)**gamma

img_data = np.minimum(255*((img_data/255)**lambda_), 255)
print('4444', img_data.shape, np.max(img_data), np.min(img_data))
cv2.imshow('e', np.uint8(img_data))
cv2.waitKey()
# save_path = r'F:\data_analysis\illumination_contrast_balancing\band_3_final.png'
save_path = r'F:\data_analysis\illumination_contrast_balancing\test4_final.tif'
cv2.imwrite(save_path, img_data)
