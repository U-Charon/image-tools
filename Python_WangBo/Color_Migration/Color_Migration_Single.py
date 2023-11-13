import time

import cv2
import numpy as np


def local_mean_function(input_data, p=0.1, overlap=0.2):
    """
    Parameters
    ----------
    p = 0.1  # 论文eg:0.1
    overlap = 0.2
    """
    h, w, c = input_data.shape  # [h, w, c]
    print( h, w)

    im_mean = np.mean(input_data)
    im_std = np.std(input_data)

    constant = 128 / 45  # 理想状态下的mean/std
    rho = p / im_std * im_mean / constant  # std越大，ρ越小， ADWs的size小

    adw_size = int(np.sqrt(rho * h * rho * w) / 2) * 2 + 1  # 奇数
    adw_stride = int(adw_size * (1 - overlap)/2)*2  # 偶数

    print(f'adw_size:{adw_size}, adw_stride{adw_stride}')

    num_h = int(((h - adw_size) / adw_stride) + 1) + 1  # 取整后考虑剩余部分
    num_w = int(((w - adw_size) / adw_stride) + 1) + 1

    # 剩余部分padding填充之后 再+adw_stride 但不能确保adw的中心点落在原img的外围
    padding_h = adw_size + adw_stride * (num_h - 1) - h + adw_stride
    padding_w = adw_size + adw_stride * (num_w - 1) - w + adw_stride
    num_h += 1  # padding增加了adw_stride
    num_w += 1

    padding_top = int(padding_h / 2)
    padding_bottom = padding_h - padding_top
    padding_left = int(padding_w / 2)
    padding_right = padding_w - padding_left
    if padding_top <= int(adw_size / 2):  # 保证边缘adw的中心点 落在原img的外围
        padding_top += int(adw_stride / 2)
        padding_bottom += int(adw_stride / 2)
        num_h += 1
    if padding_left <= int(adw_size / 2):
        padding_left += int(adw_stride / 2)
        padding_right += int(adw_stride / 2)
        num_w += 1
    # print('000', padding_top, padding_bottom, padding_left, padding_right)

    img_padding = np.pad(input_data,
                         ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
                         mode='symmetric')

    local_mean_map = np.zeros(shape=(num_h, num_w, 3), dtype=np.float32)  # num_h+1, num_w+1 49+1 72+1

    for m in range(num_h):
        adw_top = m * adw_stride
        adw_bottom = adw_top + adw_size
        for n in range(num_w):
            adw_left = n * adw_stride
            adw_right = adw_left + adw_size
            # cv2.imshow(f'map_{m, n}', img_padding[adw_top:adw_bottom, adw_left:adw_right, :])
            # cv2.waitKey()            
            # temp_path = f'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\S{m, n}.tif'
            # temp_data = np.array(img_padding[adw_top:adw_bottom, adw_left:adw_right, :]*255, dtype=np.uint8)
            # cv2.imwrite(temp_path, temp_data)
            local_mean_map[m, n] = np.mean(img_padding[adw_top:adw_bottom, adw_left:adw_right, :], axis=(0, 1))

    pad_h, pad_w, _ = img_padding.shape
    m_h = pad_h - (adw_size - 1)
    m_w = pad_w - (adw_size - 1)

    m_hh = np.reshape(np.arange(0, m_h), newshape=(1, m_h))
    m_ww = np.reshape(np.arange(0, m_w), newshape=(m_w, 1))
    idx = np.transpose(np.array(np.meshgrid(m_hh, m_ww)), axes=(2, 1, 0))

    # scale = adw_stride
    src_idx = idx / adw_stride

    src_idx = src_idx[padding_top - int(adw_size / 2): m_h - (padding_bottom - int(adw_size / 2)),
                      padding_left - int(adw_size / 2): m_w - (padding_right - int(adw_size / 2))]
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

    mean_map = (1 - u[:, :, :1]) * (1 - u[:, :, 1:]) * local_mean_x0y0 + \
               u[:, :, :1] * (1 - u[:, :, 1:]) * local_mean_x1y0 + \
               (1 - u[:, :, :1]) * u[:, :, 1:] * local_mean_x0y1 + \
               u[:, :, :1] * u[:, :, 1:] * local_mean_x1y1

    return mean_map


def polynomial_fitting(target_data, p=0.1, overlap=0.2, method=1):
    """
    Parameters
    ----------
    p = 0.1  # 论文eg:0.1
    overlap = 0.2
    method ∈ {1, 2, 3}  对应{一阶拟合， 二阶拟合， 三阶拟合}
    """
    h, w, c = target_data.shape  # [h, w, c]

    im_mean = np.mean(target_data)
    im_std = np.std(target_data)

    constant = 128 / 45  # 理想状态下的mean/std
    rho = p / im_std * im_mean / constant  # std越大，ρ越小， ADWs的size小

    adw_size = int(np.sqrt(rho * h * rho * w) / 2) * 2 + 1  # 奇数
    adw_stride = int(adw_size * (1 - overlap) / 2) * 2  # 偶数

    num_h = int(((h - adw_size) / adw_stride) + 1) + 1  # 取整后考虑剩余部分
    num_w = int(((w - adw_size) / adw_stride) + 1) + 1

    # 剩余部分padding填充之后 再补adw_stride 但不能确保adw的中心点落在原img的外围
    padding_h = adw_size + adw_stride * (num_h - 1) - h + adw_stride
    padding_w = adw_size + adw_stride * (num_w - 1) - w + adw_stride
    num_h += 1
    num_w += 1

    padding_top = int(padding_h / 2)
    padding_bottom = padding_h - padding_top
    padding_left = int(padding_w / 2)
    padding_right = padding_w - padding_left
    if padding_top < int(adw_size / 2):  # 保证边缘adw的中心点 落在原img的外围
        padding_top += int(adw_stride / 2)
        padding_bottom += int(adw_stride / 2)
        num_h += 1
    if padding_left < int(adw_size / 2):
        padding_left += int(adw_stride / 2)
        padding_right += int(adw_stride / 2)
        num_w += 1

    img_padding = np.pad(target_data,
                         ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
                         mode='reflect')

    local_mean_map = np.zeros(shape=(num_h, num_w, 3), dtype=np.float32)  # num_h+1, num_w+1 49+1 72+1
    local_mean_center = np.zeros(shape=(num_h, num_w, 2), dtype=np.uint64)  # 记录中心点位置

    for m in range(num_h):
        adw_top = m * adw_stride
        adw_bottom = adw_top + adw_size
        center_h = adw_top + int(adw_size / 2) + 1
        for n in range(num_w):
            adw_left = n * adw_stride
            adw_right = adw_left + adw_size
            center_w = adw_left + int(adw_size / 2) + 1
            local_mean_map[m, n] = np.mean(img_padding[adw_top:adw_bottom, adw_left:adw_right, :], axis=(0, 1))

            local_mean_center[m, n, :] = center_h, center_w

    # return local_mean_map, local_mean_center
    """Polynomial Fitting"""
    lmm = local_mean_map[1:-1, 1:-1]
    lmc = local_mean_center[1:-1, 1:-1] - np.array([padding_top, padding_left])  # 除去padding对应与原图的真实坐标

    """构造target_surface"""
    t_y = np.arange(0, h)
    t_x = np.arange(0, w)
    t_idx = np.transpose(np.array(np.meshgrid(t_y, t_x), dtype=np.uint64), axes=(2, 1, 0))  # [h, w, 2]

    """order method = 1, 2, 3"""

    if method >= 1:
        x1 = np.reshape(lmc[..., 0], newshape=(-1, 1))  # i
        x2 = np.reshape(lmc[..., 1], newshape=(-1, 1))  # j
        x0 = np.ones_like(x1)  # 1
        x = np.concatenate([x0, x1, x2], axis=1)

        x_1 = np.reshape(t_idx[..., 0], newshape=(-1, 1))  # i
        x_2 = np.reshape(t_idx[..., 1], newshape=(-1, 1))  # j
        x_0 = np.ones_like(x_1)  # 1
        xx = np.concatenate([x_0, x_1, x_2], axis=1)

        if method >= 2:
            x3 = x1 ** 2  # i**2
            x4 = x2 ** 2  # j**2
            x5 = x1 * x2  # i*j
            x = np.concatenate([x, x3, x4, x5], axis=1)

            x_3 = x_1 ** 2  # i**2
            x_4 = x_2 ** 2  # j**2
            x_5 = x_1 * x_2  # i*j
            xx = np.concatenate([xx, x_3, x_4, x_5], axis=1)

            if method >= 3:
                x6 = x1 ** 3  # i**3
                x7 = x2 ** 3  # j**3
                x8 = x1 * x4  # i**2*j
                x9 = x2 * x3  # j**2*i
                x = np.concatenate([x, x6, x7, x8, x9], axis=1)

                x_6 = x_1 ** 3  # i**3
                x_7 = x_2 ** 3  # j**3
                x_8 = x_1 * x_4  # i**2 *j
                x_9 = x_2 * x_3  # j**2 *i
                xx = np.concatenate([xx, x_6, x_7, x_8, x_9], axis=1)
    fitting_name = ['first_order', 'second_order', 'Third_order']
    print(f"{fitting_name[method-1]} 构造的 surface")

    # y ~ f(x) = x · alpha ==> [n, 3] = [n, (m+2)*(m+1)/2] · [(m+2)*(m+1)/2 3]
    # (1, i, j)的m阶拟合 其项数关系满足：3+n-1 选 3-1个的组合
    y = np.reshape(lmm, newshape=(-1, 3))  # [n, 3]

    # # 多元线性回归求解 alpha
    # xtx = np.dot(x.T, x)
    # xty = np.dot(x.T, y)
    # alpha = np.linalg.solve(xtx, xty)

    # 根据最小二乘法拟合的多项式 系数alpha = inv(X'·X)·X'·y
    alpha = np.linalg.multi_dot([np.linalg.inv(x.T.dot(x)), x.T, y])  # alpha.shape (10, 3)

    # 求解target_surface = xx · alpha
    target_surface = np.reshape(np.dot(xx, alpha), newshape=(h, w, c))
    return target_surface


def gamma_correction(input_data, input_p, input_overlap,
                     target_data, target_p, target_overlap,
                     alpha, method=0):
    """
    Parameters
    ----------
    input_data: 需要归一化到[0, 1],float
    input_p: 论文eg: p=0.1
    input_overlap: 论文eg: overlap=0.2

    target_data: 需要归一化到[0, 1],float
    target_p: 论文eg, p=0.1
    target_overlap: 论文eg: overlap=0.2

    alpha: ∈[0, 1] 整体亮度 eg:1
    method: ∈ {0, 1, 2, 3, 4}  对应 {Single_color, First_order, Second_order, Third_order, Bi_linear_interpolation}
    """
    local_mean_map = local_mean_function(input_data, input_p, input_overlap)
    if method == 0:
        target_color_map = np.mean(target_data, axis=(0, 1))
        print('Single_color 构造的 surface')
    elif method == 4:
        target_color_map = local_mean_function(target_data, target_p, target_overlap)
        print('Bi_linear_interpolation 构造的 surface')
    else:
        target_color_map = polynomial_fitting(target_data, target_p, target_overlap, method)

    gamma = np.log(target_color_map)/np.log(local_mean_map)
    out_data = alpha * input_data ** gamma
    return out_data


if __name__ == '__main__':
    import time

    """单张影像的色彩映射"""
    # s_path = r"F:\data_analysis\Color_Migration\Color_Migration_single\LWsimg.png"
    # r_path = r"F:\data_analysis\Color_Migration\Color_Migration_single\LWrimg.png"

    # # s_path = r"F:\data_analysis\Color_Migration\Color_Migration_single\FB_simg.png"
    # # r_path = r"F:\data_analysis\Color_Migration\Color_Migration_single\FB_rimg.png"

    # s_data = cv2.imread(s_path, -1)[..., :3]
    # r_data = cv2.imread(r_path, -1)[..., :3]
    # the_method = 4  # 每个方法 保存结果不覆盖
    # t1 = time.time()
    # output_data = gamma_correction(input_data=s_data / 255, input_p=0.05, input_overlap=0.2,
    #                                target_data=r_data / 255, target_p=0.05, target_overlap=0.2,
    #                                alpha=1, method=the_method)
    # t2 = time.time()
    # print(f'总耗时： {t2-t1}')
    # out_path = fr"F:\data_analysis\Color_Migration\Color_Migration_single\LWout1{the_method}.png"
    # # out_path = fr"F:\data_analysis\Color_Migration\Color_Migration_single\FB_out_05_{the_method}.png"
    # cv2.imwrite(out_path, np.uint8(output_data * 255))

    """多幅图像分别直接色彩映射、再拼接"""
    from pathlib import Path
    import os
    def duozhang(source_dir, reference_dir, out_dir, input_p=0.1, target_p=0.1):
        source_dir = Path(source_dir)
        reference_dir = Path(reference_dir)
        out_dir = Path(out_dir)

        if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        source_paths = list(source_dir.glob('*.tif'))
        reference_paths = list(reference_dir.glob('*.tif'))
        # out_paths = list(out_dir.glob('*.tif'))

        for i in range(len(source_paths)):
            print(i)
            s_path = source_paths[i]
            r_path = reference_paths[i]
            # print(s_path, r_path)
            # m = s_path.name.split("__")[0][-1]
            # n = s_path.name.split("__")[1][0]
            # print(m,n)
            
            simg_data = cv2.imread(str(s_path), -1)
            # cv2.imshow(f'simg_{m}__{n}', simg_data)

            rimg_data = cv2.imread(str(r_path), -1)
            # cv2.imshow(f'rimg_{m}__{n}', rimg_data)

            the_method = 4  # 每个方法 保存结果不覆盖
            t1 = time.time()
            output_data = gamma_correction(input_data=simg_data / 255, input_p=input_p, input_overlap=0.2,
                                        target_data=rimg_data / 255, target_p=target_p, target_overlap=0.2,
                                        alpha=1, method=the_method)
            t2 = time.time()
            print(f'单张总耗时： {t2-t1}')

            # cv2.imshow(f'out_{m}__{n}', np.uint8(output_data*255))
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # out_path = out_dir / f"out_{m}__{n}.tif"
            out_path = out_dir / r_path.name
            cv2.imwrite(str(out_path), np.uint8(output_data * 255))

    """example_1"""
    # source_dir = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\s_img'
    # reference_dir = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\r_img'
    # out_dir = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\out_img_single'

    """example_big"""
    # source_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_source"
    # reference_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_reference"
    # out_dir = r"F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_out_single"

    """example_small"""
    # source_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\small\simg_v2"
    # reference_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\small\rimg"
    # out_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\small\out_single"

    """LW"""
    # source_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\simg"
    # reference_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\rimg"
    # out_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\out_single"

    """ancheng:h30-->h29"""
    source_dir = r"F:\Dataset_Aerial\h30"
    reference_dir = r"F:\Dataset_Aerial\JP_AnCheng_h29\01_sources\0_img"
    out_dir = r"F:\Dataset_Aerial\h30_to_h29"

    duozhang(source_dir, reference_dir, out_dir, 0.05, 0.05)

