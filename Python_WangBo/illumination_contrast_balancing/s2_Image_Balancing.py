import cv2
import numpy as np

# img_path = r'F:\data_analysis\illumination_contrast_balancing\band_3_elim.png'
img_path = r'F:\data_analysis\illumination_contrast_balancing\test2_elim.tif'
img_data = cv2.imread(img_path, -1)

cv2.imshow('light_elim', img_data)

h, w, c = img_data.shape

blk_size = 61
epsilon = 1e-8


def DE(img_ref_block):
    """
    计算 Definition：文中解释卫清晰度
    """
    M, N, _ = img_ref_block.shape
    delta_x = img_ref_block[1:, ...] - img_ref_block[:-1, ...]  # [M-1, N, _]
    delta_y = img_ref_block[:, 1:, ...] - img_ref_block[:, :-1, ...]  # [M, N-1, _]

    return np.sum(np.sqrt((delta_x[:, 1:N] ** 2 + delta_y[1:M, :] ** 2) / 2), axis=(0, 1, 2)) / ((M - 1) * (N - 1) + epsilon)


'''遍历 block_size '''

h_num_step = h // blk_size + 1
w_num_step = w // blk_size + 1

all_blocks = np.zeros(shape=(h_num_step, w_num_step, 1 + 3 * 2), dtype=np.float32)

for i in range(h_num_step):
    h_start = i * blk_size
    h_end = (i + 1) * blk_size
    if (i + 1) * blk_size > h:
        h_end = h
    for j in range(w_num_step):
        w_start = j * blk_size
        w_end = (j + 1) * blk_size
        if (j + 1) * blk_size > w:
            w_end = w

        img_block = img_data[h_start:h_end, w_start:w_end]
        all_blocks[i, j, 0] = DE(img_block)
        all_blocks[i, j, 1:4] = np.mean(img_block, axis=(0, 1))
        all_blocks[i, j, 4:7] = np.std(img_block, axis=(0, 1))

ref_index = np.unravel_index(np.argmax(all_blocks[..., 0]), all_blocks[..., 0].shape)  # 根据最大De值 找到指定ref_Block
print(ref_index)
ref_block = img_data[ref_index[0]:ref_index[0] + blk_size, ref_index[1]:ref_index[1] + blk_size]
cv2.imshow('00', ref_block)
Mean_ref = all_blocks[ref_index][1:4]
Std_ref = all_blocks[ref_index][4:7]
print(Mean_ref)
print(Std_ref)

"""
逐像素计算以blk_size为领域的Mean Std
"""
Mean_nbr = np.zeros_like(img_data, dtype=np.float32)
Std_nbr = np.zeros_like(img_data, dtype=np.float32)
half_blk_size = blk_size // 2
for hi in range(h):
    if hi - half_blk_size >= 0:
        h_top = hi - half_blk_size
    else:
        h_top = 0

    if hi + half_blk_size + 1 >= h:
        h_bottom = h
    else:
        h_bottom = hi + half_blk_size + 1

    for wi in range(w):
        if wi - half_blk_size >= 0:
            w_lift = wi - half_blk_size
        else:
            w_lift = 0

        if wi + half_blk_size + 1 >= w:
            w_right = w
        else:
            w_right = wi + half_blk_size + 1

        Mean_nbr[hi, wi, ...] = np.mean(img_data[h_top: h_bottom, w_lift: w_right, ...], axis=(0, 1))
        Std_nbr[hi, wi, ...] = np.std(img_data[h_top: h_bottom, w_lift: w_right, ...], axis=(0, 1))

print(Mean_nbr.shape)
print('Mean_nbr', np.mean(Mean_nbr, axis=(0, 1)))
print(Std_nbr.shape)
print('Std_nbr', np.mean(Std_nbr, axis=(0, 1)))

w_s = Std_ref / (Std_ref + Std_nbr)
w_m = Mean_ref / (Mean_ref + Mean_nbr)


alpha = w_s * Std_ref / (w_s * Std_nbr + (1 - w_s) * Std_ref)
beta = w_m * Mean_ref + (1 - w_m - alpha) * Mean_nbr

result = alpha * img_data + beta
print(result.dtype, np.max(result), np.min(result))

# result = (result-np.min(result))/(np.max(result)-np.min(result))
cv2.imshow('image_balancing', np.uint8(result))
# save_path = r'F:\data_analysis\illumination_contrast_balancing\band_3_balancing.png'
# save_path = r'F:\data_analysis\illumination_contrast_balancing\test2_balancing.tif'
# cv2.imwrite(save_path, np.uint8(result))
cv2.waitKey()
