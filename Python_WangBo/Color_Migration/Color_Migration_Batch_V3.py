# encoding=utf-8
'''
1. 根据.tfw找对应的八邻域影像   
2. 正对非拼凑型数据数据构成非矩形的异形结构, 直接指定adw_size

        x'= Ax + By + C       
        y'= Dx + Ey + F
            #   x': 象素对应的地理X坐标
            #   y': 象素对应的地理Y坐标
            #   x: 象素坐标【列号】
            #   y: 象素坐标【行号】
        .tfw文件说明:
            #   A: X方向上的象素分辨率
            #   D: 旋转系统
            #   B: 旋转系统
            #   E: Y方向上的象素分辨素
            #   C: 栅格地图左上角象素中心X坐标
            #   F: 栅格地图左上角象素中心Y坐标
                ArcGIS中坐标的原点是在左下角, TIFF文件的坐标计算是在左上角
'''

import os

import cv2
import numpy as np

from pathlib import Path


def global_adw(base_dir:Path, source_dir:Path, p=0.1, overlap=0.2):
    # base
    base_dir = Path(base_dir)
    base_tfw = [_ for _ in base_dir.glob('*.tfw')]
    base_pth = [_ for _ in base_dir.glob('*.tif')]
    assert len(base_tfw)==1 and len(base_pth)==1, "base只需要一张底图"
    base_txt = base_tfw[0].read_text()
    base_A, base_D, base_B, base_E, base_C, base_F = [float(_) for _ in base_txt.split()]
    base_affine = np.array([[base_A, base_D],[base_B, base_E]])
    base_bias = np.array([base_C, base_F])
    print(f'base: A = {base_affine}\nbias = {base_bias}')

    source_dir = Path(source_dir)
    source_tfws = [_ for _ in source_dir.glob('*.tfw')]
    position=[]
    source_Sta = np.zeros(shape=(len(source_tfws), 3))  # "3"分别对应：mean var pix_num
    for i, s_tfw in enumerate(source_tfws):
        source_txt = s_tfw.read_text()
        source_A, source_D, source_B, source_E, source_C, source_F = [float(_) for _ in source_txt.split()] 
        source_affine = np.matrix([[source_A, source_D],[source_B, source_E]])
        source_bias = np.array([source_C, source_F])
        # print(f'source: A = {source_affine}\nbias = {source_bias}')
        # 单张source影像像素值统计、记录
        source_pth = source_dir / (str(s_tfw.name).split('.')[0] + '.tif')
        source_data = cv2.imread(str(source_pth))[..., :3]
        source_Sta[i, 0] = np.mean(source_data)  # 组内均值
        source_Sta[i, 1] = np.sum((np.mean(source_data) - source_data)**2)  # 组内方差
        h, w, _ = source_data.shape
        source_Sta[i, 2] = h * w * 3  # 样本量

        # 记录图像位置
        position.append((str(s_tfw.name).split('.')[0], float(source_C), float(source_F)))


    # 图像位置
    data_type = [('name', "U100"), ('C', float), ('F', float)]  # U100 代表str100
    position = np.array(position, dtype=data_type)
    num_cols = len(np.unique(position['C']))  # 横向img个数
    num_rows = len(np.unique(position['F']))  # 纵向img个数
    position = np.sort(position, order=['F', 'C'], )  # 先F后C 进行升序排列
    position_table = np.reshape(position, newshape=(num_rows, num_cols))
    position_table = np.flip(position_table, axis=0)  # 由于F是实际顺序是降序，因此在reshape之后需要在对位置进行上下翻转
    del position
    print(f'position_table: {position_table}')


    # source统计
    sources_mean = np.sum(source_Sta[:, 0] * source_Sta[:, 2]) / np.sum(source_Sta[:, 2])
    source_SSA = np.sum((source_Sta[:, 0] - sources_mean)**2 * source_Sta[:, 2])  # 组间方差
    source_SSE = np.sum(source_Sta[:, 1])
    source_SST = source_SSA + source_SSE  # 总方差
    sources_std = np.sqrt(source_SST / np.sum(source_Sta[:, 2]))
    del source_Sta, source_SSA, source_SSE, source_SST
    print(f'source mean: {sources_mean}, std: {sources_std}')

    constant = 128 / 45  # 理想状态下的mean/std
    rho = p / sources_std * sources_mean / constant  # std越大，ρ越小， ADWs的size小

    # 计算大图的total_w, total_h
    total_w = 0
    each_w = []
    for i in range(num_cols):
        temp_data = cv2.imread(str(source_dir / (position_table[0, i]['name']+'.tif')), -1)
        _, temp_w, _ = temp_data.shape
        total_w += temp_w
        each_w.append(temp_w)
    del temp_data, temp_w

    total_h = 0
    each_h = []
    for i in range(num_rows):
        temp_data = cv2.imread(str(source_dir / (position_table[i, 0]['name']+'.tif')), -1)
        temp_h, _, _ = temp_data.shape
        total_h += temp_h
        each_h.append(temp_h)
    del temp_data, temp_h

    # 计算suorce 的adw_size, adw_stride
    source_adw_size = int(np.sqrt(rho * total_w * rho * total_h) / 2) * 2 + 1  # 奇数
    source_adw_stride = int(source_adw_size * (1 - overlap) / 2) * 2  # 偶数
    print(f'source adw  size: {source_adw_size}, stride: {source_adw_stride}')


    # 根据仿射矩阵计算source的经纬度范围 并转为 base的像素坐标范围
    base_start = np.array([position_table[0, 0]['C'], position_table[0, 0]['F']])
    base_pix_start = (base_start-base_bias).dot(np.linalg.inv(base_affine))
    base_x, base_y = int(base_pix_start[0]), int(base_pix_start[1])
    
    base_size = np.array([total_w, total_h]).dot(source_affine).dot(np.linalg.inv(base_affine))
    base_w, base_h = int(base_size[0, 0]), int(base_size[0, 1])

    # base 统计对应source范围内的mean,std
    base_data = cv2.imread(str(base_pth[0]), -1)[base_y:base_y+base_h:, base_x:base_x+base_w, :3]   # 整个色迁过程只考虑 RGB
    # cv2.imshow('base', base_data)
    # cv2.waitKey()
    base_mean = np.mean(base_data)
    base_std = np.std(base_data)
    base_rho = p / base_std * base_mean / constant  # std越大，ρ越小， ADWs的size小
    # 计算suorce 的adw_size, adw_stride    
    base_adw_size = int(np.sqrt(base_rho * total_w * base_rho * total_h) / 2) * 2 + 1  # 奇数
    base_adw_stride = int(base_adw_size * (1 - overlap) / 2) * 2  # 偶数
    print(f'base adw  size: {base_adw_size}, stride: {base_adw_stride}')

    return position_table, source_adw_size, source_adw_stride, base_adw_size, base_adw_stride, each_w, each_h


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


    padding_row = np.zeros(shape=(len(each_h), 2), dtype=np.int32)
    for row in range(len(each_h)):
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
  

    padding_col = np.zeros(shape=(len(each_w), 2), dtype=np.int32)
    for col in range(len(each_w)):
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


# def local_mean_function()


def base_imgs(base_dir, source_dir):
    base_dir = Path(base_dir)
    base_tfw = [_ for _ in base_dir.glob('*.tfw')]
    base_img = [_ for _ in base_dir.glob('*.tif')]
    assert len(base_tfw)==1 and len(base_img)==1, "base只需要一张底图"
    base_tfw, base_img = base_tfw[0], base_img[0]
    base_txt = base_tfw.read_text()
    base_A, base_D, base_B, base_E, base_C, base_F = [float(_) for _ in base_txt.split()]
    print(base_A, base_D, base_B, base_E, base_C, base_F )
    base_affine = np.array([[base_A, base_D],[base_B, base_E]])
    base_bias = np.array([base_C, base_F])
    # croods' = croods * affine + bias 矩阵关系
    print(base_affine, base_bias)  # 显示时有精度损失，base_affine[0, 0], base_bias[0]实际读取时没有精度损失
    # del base_A, base_D, base_B, base_E, base_C, base_F

    base_data = cv2.imread(str(base_img), -1)
    print(base_data.shape)

    source_dir = Path(source_dir)
    source_tfw = source_dir.glob('*.tfw')
    source_img = source_dir.glob('*.tif')
    for s_tfw, s_pth in zip(source_tfw, source_img):
        source_txt = s_tfw.read_text()
        source_A, source_D, source_B, source_E, source_C, source_F = [float(_) for _ in source_txt.split()] 
        source_affine = np.matrix([[source_A, source_D],[source_B, source_E]])
        source_bias = np.array([source_C, source_F])
        # 根据当前source图像的.tfw以及w、h计算base对应范围的起点像素坐标以及w、h
        pix_croods = (source_bias - base_bias).dot(np.linalg.inv(base_affine))
        pix_croods = np.array(pix_croods, dtype=np.int16)
        print('起点：', pix_croods[0], pix_croods[1])
        source_img = cv2.imread(str(s_pth), -1)
        h, w, _ = source_img.shape
        b_size = np.array([h, w]).dot(source_affine).dot(np.linalg.inv(base_affine))
        b_size = np.squeeze(np.array(b_size, dtype=np.int16))
        print('[h, w]', b_size)

        # 根据 起点、h, w计算
        base_img = base_data[pix_croods[1]:pix_croods[1]+b_size[0], pix_croods[0]:pix_croods[0]+b_size[1], :]
        cv2.imshow(f'{s_pth.name}', base_img)
        cv2.waitKey()
        base_pth = base_dir / s_pth.name 
        cv2.imwrite(str(base_pth), cv2.resize(base_img, (w, h))) 
        base_tfw_ = base_dir / s_tfw.name
        base_txt_ = f'{base_A}\n{base_D}\n{base_B}\n{base_E}\n{source_C}\n{source_F}'
        base_tfw_.write_text(base_txt_)


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
    target_dir = Path(target_dir)
    position_table, source_adw_size, source_adw_stride, target_adw_size, target_adw_stride, each_w, each_h = global_adw(target_dir, source_dir, s_p, overlap) 

    source_padding_row, source_padding_col = padding_size(source_adw_size, source_adw_stride, each_h, each_w)
    print(source_padding_row, source_padding_col)
    target_padding_row, target_padding_col = padding_size(target_adw_size, target_adw_stride, each_h, each_w)
    print(target_padding_row, target_padding_col)

    for i in range(position_table.shape[0]):
        for j in range(position_table.shape[1]):
            print(position_table[i, j])
            source_path = source_dir / (position_table[0, 0]['name'] + '.tif') 

            # t_mean_map = local_mean_function(target_adw_size, target_adw_stride, target_path, target_padding_row, target_padding_col)
            # s_mean_map = local_mean_function(source_adw_size, source_adw_stride, source_path, source_padding_row, source_padding_col)

            # gamma = np.log(t_mean_map) / np.log(s_mean_map)
 
            o_img_data = cv2.imread(str(source_path), -1)/255
            print(o_img_data.shape)
            # out_data = alpha * o_img_data**gamma

            # cv2.imshow(f'out_data{img_row}_{img_col}', np.array(out_data))
            # cv2.waitKey()

            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # save_path = os.path.join(output_dir, 'out_' + str(img_row) + '__' + str(img_col) + '.tif')  # 暂时source写死
            # cv2.imwrite(save_path, np.array(out_data * 255, dtype=np.uint8))



    




if __name__ == '__main__':
    # tfw_dir = r'F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\source'
    # img_table = building_positional(tfw_dir)
    # print(img_table[0, 10])
    # print(img_table[1, 8])

    base_dir = r'F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\base'
    source_dir = r'F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\source'
    # base_imgs(base_dir, source_dir)



    # global_adw(base_dir, source_dir)

    base_dir = r'F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\base'
    source_dir = r'F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\source'
    output_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\out_01_01"
    batch_gamma_correction(source_dir, base_dir, output_dir, alpha=1, s_p=0.01, r_p=0.01, overlap=0.2)


    # 实验
    # img_pt = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\source\J50H001073.tif"
    # img_data = cv2.imread(img_pt,-1)
    # h, w, c = img_data.shape
    # print(h, w, c)