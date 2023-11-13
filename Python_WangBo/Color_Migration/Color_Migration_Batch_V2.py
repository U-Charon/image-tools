import os

import cv2
import numpy as np

from pathlib import Path


def global_adw(img_dir, p=0.1, overlap=0.2):
    origin_img_paths = list(img_dir.glob('*.tif'))

    img_row_max = 0  # img row num
    img_col_max = 0  # img col num
    imgs_Sta = np.zeros(shape=(len(origin_img_paths), 3))  # "3"分别对应：mean var pix_num

    for i, o_img_path in enumerate(origin_img_paths):
        img_position = str(o_img_path).split('__')

        img_row = int(img_position[0][-1])  # row
        if img_row > img_row_max:
            img_row_max = img_row

        img_col = int(img_position[1][0])  # col
        if img_col > img_col_max:
            img_col_max = img_col

        o_img_data = cv2.imread(str(o_img_path), -1) / 255
        imgs_Sta[i, 0] = np.mean(o_img_data)  # 组内均值
        # imgs_Sta[i, 1] = np.var(o_img_data)  # 组内方差
        imgs_Sta[i, 1] = np.sum((np.mean(o_img_data) - o_img_data)**2)  # 组内方差
        imgs_Sta[i, 2] = o_img_data.shape[0] * o_img_data.shape[1] * o_img_data.shape[2]  # 样本量

        # cv2.imshow('origin', o_img_data)
        # cv2.waitKey()

    imgs_mean = np.sum(imgs_Sta[:, 0] * imgs_Sta[:, 2]) / np.sum(imgs_Sta[:, 2])

    imgs_SSA = np.sum((imgs_Sta[:, 0] - imgs_mean)**2 * imgs_Sta[:, 2])  # 组间方差
    # imgs_SSE = np.sum(imgs_Sta[:, 1] * imgs_Sta[:, 2]) # 组内方差
    imgs_SSE = np.sum(imgs_Sta[:, 1])
    imgs_SST = imgs_SSA + imgs_SSE  # 总方差
    imgs_std = np.sqrt(imgs_SST / np.sum(imgs_Sta[:, 2]))
    del imgs_Sta, imgs_SSA, imgs_SSE, imgs_SST

    constant = 128 / 45  # 理想状态下的mean/std
    rho = p / imgs_std * imgs_mean / constant  # std越大，ρ越小， ADWs的size小

    total_h = 0
    total_w = 0
    prefix = (origin_img_paths[0].name).split("_")[0]
    for i in range(1, img_row_max + 1):
        first_col_img_path = img_dir / (prefix + '_' + str(i) + '__' + str(1) + '.tif')
        first_col_img_data = cv2.imread(str(first_col_img_path), -1)
        total_h += first_col_img_data.shape[0]
    for j in range(1, img_col_max + 1):
        first_row_img_path = img_dir / (prefix + '_' + str(1) + '__' + str(j) + '.tif')
        first_row_img_data = cv2.imread(str(first_row_img_path), -1)
        total_w += first_row_img_data.shape[1]

    adw_size = int(np.sqrt(rho * total_h * rho * total_w) / 2) * 2 + 1  # 奇数
    adw_stride = int(adw_size * (1 - overlap) / 2) * 2  # 偶数

    return adw_size, adw_stride

def padding_size(adw_size, adw_stride, each_h, each_w):
    total_h = np.sum(each_h)
    total_w = np.sum(each_w)
    num_h = int(((total_h - adw_size) / adw_stride) + 1) + 1  # 取整后考虑剩余部分
    num_w = int(((total_w - adw_size) / adw_stride) + 1) + 1  

    # 剩余部分padding填充之后 再+adw_stride 但不能确保adw的中心点落在原img的外围
    padding_h = adw_size + adw_stride * (num_h - 1) - total_h + adw_stride
    padding_w = adw_size + adw_stride * (num_w - 1) - total_w + adw_stride
    num_h += 1  # padding增加了adw_stride
    num_w += 1

    padding_top = int(padding_h / 2)
    # padding_bottom = padding_h - padding_top
    if padding_top <= int(adw_size / 2):  # 保证边缘adw的中心点 落在原img的外围
        padding_top += int(adw_stride / 2)
        # padding_bottom += int(adw_stride / 2)
        num_h += 1

    padding_left = int(padding_w / 2)
    # padding_right = padding_w - padding_left
    if padding_left <= int(adw_size / 2):  # 保证边缘adw的中心点 落在原img的外围
        padding_left += int(adw_stride / 2)
        # padding_right += int(adw_stride / 2)
        num_w += 1


    padding_row = np.zeros(shape=(img_row_max, 2), dtype=np.int32)
    for row in range(img_row_max):
        # 计算 padding_bottom
        each_num_h = int((padding_top + each_h[row] - adw_size) / adw_stride + 1) + 1
        padding_bottom = adw_size + adw_stride * (each_num_h - 1) - (each_h[row] + padding_top)
        while padding_bottom <= int(adw_size/2):
            padding_bottom += adw_stride

        # 记录 padding_top padding_bottom
        padding_row[row, 0] = int(padding_top)
        padding_row[row, 1] = int(padding_bottom)

        # 更新 padding_top
        padding_top = adw_size + adw_stride - padding_bottom
  

    padding_col = np.zeros(shape=(img_col_max, 2), dtype=np.int32)
    for col in range(img_col_max):
        # 计算 padding_right
        each_num_w = int((padding_left + each_w[col] - adw_size) / adw_stride + 1) + 1
        padding_right = adw_size + adw_stride * (each_num_w - 1) - (each_w[col] + padding_left)
        while padding_right <= int(adw_size/2):
            padding_right += adw_stride

        # 记录 padding_left, padding_right
        padding_col[col, 0] = int(padding_left)
        padding_col[col, 1] = int(padding_right)

        # 更新padding_left
        padding_left = adw_size + adw_stride - padding_right

    return padding_row, padding_col


def local_mean_function(adw_size, adw_stride, img_path, padding_row, padding_col):
    img_position = str(img_path).split('__')
    global img_row, img_col
    img_row = int(img_position[0][-1])  # row
    img_col = int(img_position[1][0])  # col

    img_data = cv2.imread(str(img_path), -1) / 255
    h, w, _ = img_data.shape

    img_top_path = img_position[0][:-1] + str(img_row-1) + "__" + str(img_col) + img_position[1][1:]
    if os.path.exists(img_top_path):
        img_top_data = cv2.imread(str(img_top_path), -1) / 255
    else:
        img_top_data = cv2.flip(img_data, 0)  # 上下翻转

    img_bottom_path = img_position[0][:-1] + str(img_row + 1) + '__' + str(img_col) + img_position[1][1:]
    if os.path.exists(img_bottom_path):
        img_bottom_data = cv2.imread(str(img_bottom_path), -1) / 255
    else:
        img_bottom_data = cv2.flip(img_data, 0)  # 上下翻转

    img_left_path = img_position[0][:-1] + str(img_row) + '__' + str(img_col - 1) + img_position[1][1:]
    if os.path.exists(img_left_path): 
        img_left_data = cv2.imread(str(img_left_path), -1) / 255
    else:
        img_left_data = cv2.flip(img_data, 1)  # 左右翻转

    img_right_path = img_position[0][:-1] + str(img_row) + '__' + str(img_col + 1) + img_position[1][1:]
    if os.path.exists(img_right_path): 
        img_right_data = cv2.imread(str(img_right_path), -1) / 255
    else:
        img_right_data = cv2.flip(img_data, 1)  # 左右翻转

    img_left_top_path = img_position[0][:-1] + str(img_row - 1) + '__' + str(img_col - 1) + img_position[1][1:]
    if os.path.exists(img_left_top_path): 
        img_left_top_data = cv2.imread(str(img_left_top_path), -1) / 255
    else:
        img_left_top_data = cv2.flip(img_data, -1)  # 上下左右翻转

    img_left_bottom_path = img_position[0][:-1] + str(img_row + 1) + '__' + str(img_col - 1) + img_position[1][1:]
    if os.path.exists(img_left_bottom_path): 
        img_left_bottom_data = cv2.imread(str(img_left_bottom_path)) / 255
    else:
        img_left_bottom_data = cv2.flip(img_data, -1)  # 上下左右翻转

    img_right_top_path = img_position[0][:-1] + str(img_row - 1) + '__' + str(img_col + 1) + img_position[1][1:]
    if os.path.exists(img_right_top_path): 
        img_right_top_data = cv2.imread(str(img_right_top_path), -1) / 255
    else:
        img_right_top_data = cv2.flip(img_data, -1)  # 上下左右翻转

    img_right_bottom_path = img_position[0][:-1] + str(img_row + 1) + '__' + str(img_col + 1) + img_position[1][1:]
    if os.path.exists(img_right_bottom_path): 
        img_right_bottom_data = cv2.imread(str(img_right_bottom_path), -1) / 255
    else:
        img_right_bottom_data = cv2.flip(img_data, -1)  # 上下左右翻转
    
    padding_top, padding_bottom = padding_row[img_row-1]
    padding_left, padding_right = padding_col[img_col-1]

    img_padding_1 = np.concatenate((img_left_top_data[-padding_top:,-padding_left:], 
                                    img_top_data[-padding_top:, :],
                                    img_right_top_data[-padding_top:, :padding_right]), axis=1)

    img_padding_2 = np.concatenate((img_left_data[:, -padding_left:], 
                                    img_data,
                                    img_right_data[:, :padding_right]), axis=1)

    img_padding_3 = np.concatenate((img_left_bottom_data[:padding_bottom, -padding_left:],
                                    img_bottom_data[:padding_bottom, :],
                                    img_right_bottom_data[:padding_bottom, :padding_right]), axis=1)

    img_padding = np.concatenate((img_padding_1, img_padding_2, img_padding_3), axis=0)
    # cv2.imshow(f'img_padding_{img_row}_{img_col}', img_padding)
    # cv2.waitKey()

    pad_h, pad_w, _ = img_padding.shape

    num_h = int((padding_top + h + padding_bottom - adw_size) / adw_stride) + 1
    num_w = int((padding_left + w + padding_right - adw_size) / adw_stride) + 1
    local_mean_map = np.zeros(shape=(num_h, num_w, 3), dtype=np.float32)  # num_h+1, num_w+1 
    
    for m in range(num_h):
        adw_top = m * adw_stride
        adw_bottom = adw_top + adw_size
        for n in range(num_w):
            adw_left = n * adw_stride
            adw_right = adw_left + adw_size
            # cv2.imshow(f'map_{m, n}', img_padding[adw_top:adw_bottom, adw_left:adw_right, :])
            # cv2.waitKey()
            # temp_path = fr'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\temp3\B3{m, n}.tif'
            # temp_data = np.array(img_padding[adw_top:adw_bottom, adw_left:adw_right, :]*255, dtype=np.uint8)
            # cv2.imwrite(temp_path, temp_data)
            # print(adw_top, adw_bottom, adw_left, adw_right)
            local_mean_map[m, n] = np.mean(img_padding[adw_top:adw_bottom, adw_left:adw_right, :], axis=(0, 1))
    # cv2.imshow('local_mean_map', local_mean_map)
    # cv2.waitKey()
    # cv2.imwrite(fr'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\temp3\mmp{img_row, img_col}.tif', local_mean_map)


    m_h = pad_h - (adw_size - 1)
    m_w = pad_w - (adw_size - 1)

    m_hh = np.reshape(np.arange(0, m_h), newshape=(1, m_h))
    m_ww = np.reshape(np.arange(0, m_w), newshape=(m_w, 1))
    idx = np.transpose(np.array(np.meshgrid(m_hh, m_ww)), axes=(2, 1, 0))
    
    src_idx = idx / adw_stride  # scale = adw_stride

    src_idx = src_idx[padding_top - int(adw_size / 2) : m_h - (padding_bottom - int(adw_size / 2)),
                      padding_left - int(adw_size / 2) : m_w - (padding_right - int(adw_size / 2))]

    # 目标像素的四连通域索引 int型
    src_x0y0 = np.array(np.floor(src_idx), dtype=np.int32)
    u = src_idx - src_x0y0  # bi_linear_weight
    src_x1y0 = src_x0y0 + np.array([1, 0], dtype=np.int32)
    src_x0y1 = src_x0y0 + np.array([0, 1], dtype=np.int32)
    src_x1y1 = src_x0y0 + np.array([1, 1], dtype=np.int32)

    local_mean_x0y0 = local_mean_map[src_x0y0[..., 0], src_x0y0[..., 1]]
    local_mean_x1y0 = local_mean_map[src_x1y0[..., 0], src_x1y0[..., 1]]
    local_mean_x0y1 = local_mean_map[src_x0y1[..., 0], src_x0y1[..., 1]]
    local_mean_x1y1 = local_mean_map[src_x1y1[..., 0], src_x1y1[..., 1]]

    # 双线插值
    mean_map = (1 - u[:, :, :1]) * (1 - u[:, :, 1:]) * local_mean_x0y0 + \
               u[:, :, :1] * (1 - u[:, :, 1:]) * local_mean_x1y0 + \
               (1 - u[:, :, :1]) * u[:, :, 1:] * local_mean_x0y1 + \
               u[:, :, :1] * u[:, :, 1:] * local_mean_x1y1

    return mean_map


def batch_gamma_correction(source_dir, target_dir, output_dir, alpha=1, s_p=0.1, r_p=0.1, overlap=0.2):
    """
    Parameters
    ----------
    source_dir: 待处理的影像文件夹
    target_dir: 待处理影像对应的底图文件夹
    output_dir: 待处理影像色迁后结果保存文件夹

    alpha: ∈[0, 1] 整体亮度 eg:1
    p: 论文eg, p=0.1
    overlap: 论文eg: overlap=0.2
    """

    source_dir = Path(source_dir)
    source_adw_size, source_adw_stride = global_adw(source_dir, s_p, overlap)
    source_padding_row, source_padding_col = padding_size(source_adw_size, source_adw_stride, each_h, each_w)

    target_dir = Path(target_dir)
    target_adw_size,  target_adw_stride  = global_adw(target_dir, r_p, overlap)
    target_padding_row, target_padding_col = padding_size(target_adw_size, target_adw_stride, each_h, each_w)

    source_img_paths = list(source_dir.glob('*.tif'))
    target_img_paths = list(target_dir.glob('*.tif'))

    for source_path, target_path in zip(source_img_paths, target_img_paths):

        t_mean_map = local_mean_function(target_adw_size, target_adw_stride, target_path, target_padding_row, target_padding_col)
        s_mean_map = local_mean_function(source_adw_size, source_adw_stride, source_path, source_padding_row, source_padding_col)
       
        gamma = np.log(t_mean_map) / np.log(s_mean_map)
 
        o_img_data = cv2.imread(str(source_path), -1)/255
        out_data = alpha * o_img_data**gamma

        # cv2.imshow(f'out_data{img_row}_{img_col}', np.array(out_data))
        # cv2.waitKey()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, 'out_' + str(img_row) + '__' + str(img_col) + '.tif')  # 暂时source写死
        cv2.imwrite(save_path, np.array(out_data * 255, dtype=np.uint8))



if __name__ == '__main__':

    # source_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_source"
    # target_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_target"
    # output_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_output"

    # batch_gamma_correction(source_dir, target_dir, output_dir, alpha=1, p=0.05, overlap=0.2)

    """big img"""
    # source_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_source"
    # reference_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_reference"
    # output_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_out"
    # batch_gamma_correction(source_dir, reference_dir, output_dir, alpha=1, p=0.1, overlap=0.2)

    """small img"""
    # source_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\small\simg_v2"
    # reference_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\small\rimg"
    # output_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\small\out01"
    # batch_gamma_correction(source_dir, reference_dir, output_dir, alpha=1, s_p=0.1, r_p=0.1, overlap=0.2)

    # s_dir = Path(source_dir)
    # for s_path in s_dir.glob("*.tif"):
    #     print(s_path)
    #     img_data = cv2.imread(str(s_path))[...,:3]
    #     print(img_data.shape)
    #     cv2.imwrite(str(s_path), img_data)

    """LW"""
    # source_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\simg"
    # reference_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\rimg"
    # output_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\out_01_01"

    # batch_gamma_correction(source_dir, reference_dir, output_dir, alpha=1, s_p=0.1, r_p=0.1, overlap=0.2)


    """big test"""
    source_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\s_img"
    reference_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\r_img"
    output_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\out_01_01"

    batch_gamma_correction(source_dir, reference_dir, output_dir, alpha=1, s_p=0.01, r_p=0.01, overlap=0.2)