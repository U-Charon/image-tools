
from pathlib import Path
import re




def name_replace(dir):
    dir = Path(dir)
    file_list = dir.iterdir()

    for element in file_list:
        # print(type(element.name))
        # element.name.split('_')
        print('*'*20)
        name_str=element.name
        name_str = re.sub('A', '1__', name_str)
        name_str = re.sub('B', '2__', name_str)
        name_str = re.sub('C', '3__', name_str)
        print(name_str)

        element.rename(dir/name_str)
        # print(element.rename(dir/name_str))


def samll_name_replace(dir):
    dir = Path(dir)
    file_list = dir.iterdir()
    for element in file_list:
        name_str = element.name
        name_str = re.sub('source', "target", name_str)
        # name_str = name_str[:9] + "_" + name_str[9:]
        # print(name_str[7], name_str[10])
        # name_str = 'ss' + name_str[:7] + name_str[10] + name_str[8:10] + name_str[7] + name_str[11:]
        # name_str = name_str[2:]
        print(name_str)
        element.rename(dir/name_str)


# 为small_source small_target 添加tfw
def small_tfw(dir, prefix="source", row=3, col=3):
    dir = Path(dir)
    for i in range(row):
        for j in range(col):

            name_str = prefix + "_" + str(i+1) + "__" + str(j+1)
            tfw_path = dir / (name_str+ ".tfw")

            img_path = dir / (name_str+ ".tif")
            img_data = cv2.imread(str(img_path), -1)
            h, w = img_data.shape[:2]

            tfw_data = open(tfw_path, mode="w")
            tfw_data.write(f"0.5\n0\n0\n-0.5\n{j*0.5*h}\n{i*(-0.5)*w}")
            tfw_data.close()



import cv2
import os
"""剪切小图，并按照行列顺序排列"""
def cut_img(img_path, row, col, save_dir):
    save_dir = Path(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name = str(img_path).split('\\')[-1].split('.')[0]
    print(img_name)

    img_data = cv2.imread(img_path, -1)[...,:3]
    h, w = img_data.shape[:2]
    h_small = int(h/row)
    w_small = int(w/col)
    print(h_small, w_small)
    for i in range(1,row+1):
        for j in range(1,col+1):
            cut_data = img_data[(i-1)*h_small:i*h_small, (j-1)*w_small:j*w_small]
            # cv2.imshow(f'{i}_{j}', small_data)
            # cv2.waitKey()
            save_path = save_dir / (img_name + f'_{i}__{j}.'+ 'tif')
            print(save_path)
            cv2.imwrite(str(save_path), cut_data)
            tfw_path = save_dir / (img_name + f'_{i}__{j}.'+ 'tfw')
            tfw_data = open(tfw_path, mode="w")
            tfw_data.write(f"0.5\n0\n0\n-0.5\n{(j-1)*0.5*w_small}\n{(i-1)*(-0.5)*h_small}")
            tfw_data.close()


def resize_img(img_dir, s=0.2):
    img_dir = Path(img_dir)
    img_list = img_dir.glob('*.tif')
    for img_path in img_list:
        img_data = cv2.imread(str(img_path))
        h, w = img_data.shape[:2]
        r_h, r_w = int(h*s), int(w*s)
        r_data = cv2.resize(img_data, (r_w, r_h))
        cv2.imshow(f'r_{s}', r_data)
        cv2.waitKey()
        save_dir = Path(str(img_dir) + f'_{s}')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), r_data)


def channal_to_3(img_dir):
    img_dir = Path(img_dir)
    img_list = img_dir.glob("*.tif")
    for img_path in img_list:
        img_data = cv2.imread(str(img_path))
        new_data = img_data[:,:,:3]
        cv2.imwrite(str(img_path), new_data)



if __name__ == '__main__':
    # source1 = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\source1'
    # target1 = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\target1'

    # name_replace(source1)
    # name_replace(target1)

    # for img_path in Path(target1).glob('*.tif'):
    #     img_data = cv2.imread(str(img_path), -1)
    #     print(img_data.shape)
    #     img_data = img_data[..., :3]
    #     print(img_data.shape)
    #     cv2.imwrite(str(img_path), img_data)

    # small_source = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\small_source'
    # small_target = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\small_target'
    # small_output = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\small_output'

    # samll_name_replace(small_source)
    # samll_name_replace(small_target)

    # small_tfw(small_source, prefix='source')
    # small_tfw(small_target, prefix='target')
    # small_tfw(small_output, prefix='out')

    # big_output = r'F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\Bantch_Color_Migration\big_output'
    # small_tfw(big_output, prefix='out')


    """更改LW_rimg.png 的尺寸"""
    # import cv2
    # lw_rimg_path = r"F:\data_analysis\Color_Migration\Color_Migration_single\LWsimg.png"
    # lw_rimg_data = cv2.imread(lw_rimg_path, -1)
    # w, h = lw_rimg_data.shape[:2]
    # print(w, h)
    # new_data = cv2.resize(lw_rimg_data, (h*5, w*5))
    # print(new_data.shape)
    # cv2.imshow('', new_data) 
    # cv2.waitKey()
    # cv2.imwrite(r"F:\data_analysis\Color_Migration\Color_Migration_single\LWsimg1.png", new_data)

    """剪切 LW_simg.png LW_rimg.png"""
    # lw_simg_path = r"F:\data_analysis\Color_Migration\Color_Migration_single\LWsimg.png"
    # lw_simg_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\simg"
    # cut_img(lw_simg_path, row=5, col=7, save_dir=lw_simg_dir)
    # lw_rimg_path = r"F:\data_analysis\Color_Migration\Color_Migration_single\LWrimg.png"
    # lw_rimg_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\rimg"
    # cut_img(lw_rimg_path, row=5, col=7, save_dir=lw_rimg_dir)

    """生成LW out的.tfw文件"""
    # out_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\LW\out"
    # small_tfw(out_dir, prefix='out',row=5, col=7)

    # out_dir = r'F:\data_analysis\Color_Migration\Color_Migration_bantch\normal\s_img'
    # small_tfw(out_dir, prefix='s_img',row=3, col=3)

    """resize Images"""
    # imgs_dir = r'F:\data_analysis\distributed_transform\itamiH29\Images'
    # # imgs_dir = r'F:\data_analysis\distributed_transform\ancheng\h29'
    # resize_img(imgs_dir, s=0.2)


    """big_test channel：4 to 3"""
    # source_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\s_img"
    # channal_to_3(source_dir)
    reference_dir = r"F:\data_analysis\Color_Migration\Color_Migration_bantch\big_test\r_img"
    channal_to_3(reference_dir)
