//
// Created by jerry on 2022/10/31.
//
#include <filesystem>
#include <iomanip>
#include "../../include/xmap_util.hpp"

int get_tfw_content(const char *filename, double *result) {
    std::ifstream ifs;
    std::string line;
    int i = 0;

    ifs.open(filename, std::ios::in);
    if (!ifs.is_open()) {
        return 1;
    }
    while (getline(ifs, line)) {
        if (!line.empty()) {
            result[i] = std::stod(line);
        }
        i += 1;
    }
    ifs.close();

    return 0;
}

vips::VImage get_3_bands_image(const char *filename) {
    auto img = vips::VImage::new_from_file(filename) / 255;
    if (img.bands() == 4) {
        return img[0].bandjoin(img[1]).bandjoin(img[2]);
    } else {
        return img;
    }
}

void global_adw_stride(const std::map<std::string, std::vector<long long>> &dict_tiff_to_base, const vips::VImage &base_img,
                       float p, float overlap, int full_h, int full_w, int img_num,
                       int *adw_size_base, int *adw_stride_base, int *adw_size_source, int *adw_stride_source) {
    vips::VImage source_img_data, base_img_data;
    int i = 0;

    using namespace arma;
    fmat images_stat_base(img_num, 3, fill::zeros);
    fmat images_stat_source(img_num, 3, fill::zeros);

    auto it = dict_tiff_to_base.begin();
    while (it != dict_tiff_to_base.end()) {
        source_img_data = vips::VImage::new_from_file(it->first.c_str()) / 255;
        base_img_data = base_img.extract_area(it->second[0], it->second[1], it->second[2], it->second[3]);
        base_img_data = base_img_data.resize((double) source_img_data.width() / it->second[2], vips::VImage::option()->
                set("vscale", (double) source_img_data.height() / it->second[3])->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255;

        images_stat_source(i, 0) = (float) source_img_data.avg();
        images_stat_base(i, 0) = (float) base_img_data.avg();

        auto temp_sum_source = (source_img_data.avg() - source_img_data).pow(2.0);
        auto temp_sum_base = (base_img_data.avg() - base_img_data).pow(2.0);
        auto img_sum_source = x_vips_to_arma_cube(temp_sum_source);
        auto img_sum_base = x_vips_to_arma_cube(temp_sum_base);
        fmat a_source = sum(conv_to<fcube>::from(img_sum_source), 2);
        fmat a_base = sum(conv_to<fcube>::from(img_sum_base), 2);
        images_stat_source(i, 1) = sum(sum(a_source)); // 组内方差
        images_stat_base(i, 1) = sum(sum(a_base)); // 组内方差

        images_stat_source(i, 2) = (float) source_img_data.width() * (float) source_img_data.height() * 3;
        images_stat_base(i, 2) = (float) base_img_data.width() * (float) base_img_data.height() * 3;
        i += 1;

        ++it;
    }

    auto images_mean_source = sum(images_stat_source.col(0) % images_stat_source.col(2)) / sum(images_stat_source.col(2));
    auto images_mean_base = sum(images_stat_base.col(0) % images_stat_base.col(2)) / sum(images_stat_base.col(2));

    auto images_SSA_source = sum(pow((images_stat_source.col(0) - images_mean_source), 2.0) % images_stat_source.col(2)); // 组间方差
    auto images_SSA_base = sum(pow((images_stat_base.col(0) - images_mean_base), 2.0) % images_stat_base.col(2)); // 组间方差

    auto images_SSE_source = sum(images_stat_source.col(1)); // 组内方差
    auto images_SSE_base = sum(images_stat_base.col(1)); // 组内方差

    auto images_SST_source = images_SSA_source + images_SSE_source; // 总方差
    auto images_SST_base = images_SSA_base + images_SSE_base; // 总方差

    auto images_std_source = sqrt(images_SST_source / sum(images_stat_source.col(2)));
    auto images_std_base = sqrt(images_SST_base / sum(images_stat_base.col(2)));

    float constant = 128.0 / 45;  //理想状态下的mean/std
    auto rho_source = p / images_std_source * images_mean_source / constant;  // std越大，ρ越小， ADWs的size小
    auto rho_base = p / images_std_base * images_mean_base / constant;  // std越大，ρ越小， ADWs的size小

    *adw_size_source = int(sqrt(rho_source * (float) full_h * rho_source * (float) full_w) / 2) * 2 + 1;  // 奇数
    *adw_size_base = int(sqrt(rho_base * (float) full_h * rho_base * (float) full_w) / 2) * 2 + 1;  // 奇数

    *adw_stride_source = int(float(*adw_size_source) * (1 - overlap) / 2) * 2;  // 偶数
    *adw_stride_base = int(float(*adw_size_base) * (1 - overlap) / 2) * 2;  // 偶数

}

void get_query_tables(const char *base_tfw_name, const char *source_dir,
                      std::map<std::string, std::string> *dict_tiffs_tfws,
                      std::map<std::string, std::string> *query_table_geo_tif,
                      std::map<std::string, std::vector<long long>> *dict_tiff_to_base,
                      std::map<std::string, std::vector<long long>> *query_table_tif_geo,
                      int *full_h, int *full_w, int *img_num) {

    double source_tfw_content[6], base_tfw_content[6];
    int scale = 10000;
    int full_height = 0, full_width = 0, images_num = 0;

    if (get_tfw_content(base_tfw_name, &base_tfw_content[0]) != 0) {
        printf("error reading base tfw file...\n");
        return;
    }

    for (const auto &entry: std::filesystem::directory_iterator(std::filesystem::u8path(source_dir))) {
        auto file_name = entry.path().string();
        auto ext = entry.path().extension();

        if (ext == ".tif" || ext == ".tiff" || ext == ".TIF" || ext == ".TIFF") {
            auto tif_img_data = vips::VImage::new_from_file(file_name.c_str());
            auto tfw_name = entry.path().parent_path().string() + "\\" + entry.path().stem().string() + ".tfw";
            /*
             * source图的tiff到配套tfw的映射关系，key和value都保存的是带目录的文件名[dict_tiffs_tfws]
             * string(tif):string(tfw)
             */
            dict_tiffs_tfws->insert(std::map<std::string, std::string>::value_type(file_name, tfw_name));
            if (get_tfw_content(tfw_name.c_str(), &source_tfw_content[0]) != 0) {
                printf("error reading source tfw file...\n");
                return;
            }
            /*
             * tiff 影像到小地图上的映射表构建[dict_tiff_to_base] string:[x, y, w, h]
             * Key：带目录的tif文件名
             * Value：[x, y, w, h], 要裁切的左上角坐标x, y, 要在地图上裁切的宽高w, h
             */
            auto zeroX = base_tfw_content[4] - base_tfw_content[0] / 2;
            auto zeroY = base_tfw_content[5] + base_tfw_content[0] / 2;
            auto leftX = (long long) ((source_tfw_content[4] - source_tfw_content[0] / 2 - zeroX) / base_tfw_content[0]);
            auto leftY = (long long) ((zeroY - (source_tfw_content[5] + source_tfw_content[0] / 2)) / base_tfw_content[0]);
            auto rightX = (long long) ((source_tfw_content[4] - source_tfw_content[0] / 2 + source_tfw_content[0] * tif_img_data.width() - zeroX) / base_tfw_content[0]);
            auto rightY = (long long) ((zeroY - (source_tfw_content[5] + source_tfw_content[0] / 2 - source_tfw_content[0] * tif_img_data.height())) / base_tfw_content[0]);
            std::vector<long long> v = {leftX, leftY, rightX - leftX, rightY - leftY};
            dict_tiff_to_base->insert(std::map<std::string, std::vector<long long>>::value_type(file_name, v));
            /*
             * 查询表1：query_table_geo_tif, string:string
             * Key: 该tif文件的坐标，用下划线拼接，如：{“500086_4071725”: xxx.tif}
             * Value: 带目录的tif文件名
             *
             * 查询表2：query_table_tif_geo, string:[geo_w, geo_h]
             * Key: 带目录的tif文件名
             * Value: 一个2元素的vector数组geo_w, geo_h : { xxx.tif：[500086, 4071725]}
             */
            auto x0 = (long long) ((source_tfw_content[4] - source_tfw_content[0] / 2) * scale);
            auto y0 = (long long) ((source_tfw_content[5] + source_tfw_content[0] / 2) * scale);
            auto xn = (long long) ((source_tfw_content[4] - source_tfw_content[0] / 2 + source_tfw_content[0] * tif_img_data.width()) * scale);
            auto yn = (long long) ((source_tfw_content[5] + source_tfw_content[0] / 2 - source_tfw_content[0] * tif_img_data.height()) * scale);

            auto geo_h = y0 - yn;
            auto geo_w = xn - x0;
            auto x0_y0 = std::to_string(x0) + "_" + std::to_string(y0);
            printf("geo_tif [%s, %s]\n", x0_y0.c_str(), file_name.c_str());
            query_table_geo_tif->insert(std::map<std::string, std::string>::value_type(x0_y0, file_name));
            std::vector<long long> v2 = {x0, y0, geo_w, geo_h};
            query_table_tif_geo->insert(std::map<std::string, std::vector<long long>>::value_type(file_name, v2));

            // 累加返回值
            images_num += 1;
            full_height += tif_img_data.height();
            full_width += tif_img_data.width();
        }
    }
    *full_h = full_height;
    *full_w = full_width;
    *img_num = images_num;
}

void padding_calculation(int h, int w, int adw_size, int adw_stride, int *top, int *bottom, int *left, int *right, int *number_h, int *number_w) {
    auto num_h = int(((h - adw_size) / adw_stride) + 1) + 1;
    auto num_w = int(((w - adw_size) / adw_stride) + 1) + 1;

    auto padding_h = adw_size + adw_stride * (num_h - 1) - h + adw_stride;
    auto padding_w = adw_size + adw_stride * (num_w - 1) - w + adw_stride;
    num_h += 1;
    num_w += 1;

    auto padding_top = int(padding_h / 2);
    auto padding_bottom = padding_h - padding_top;
    auto padding_left = int(padding_w / 2);
    auto padding_right = padding_w - padding_left;
    if (padding_top <= int(adw_size / 2)) {
        padding_top += int(adw_stride / 2);
        padding_bottom += int(adw_stride / 2);
        num_h += 1;
    }
    if (padding_left <= int(adw_size / 2)) {
        padding_left += int(adw_stride / 2);
        padding_right += int(adw_stride / 2);
        num_w += 1;
    }

    *top = padding_top;
    *bottom = padding_bottom;
    *left = padding_left;
    *right = padding_right;
    *number_h = num_h;
    *number_w = num_w;
}

vips::VImage local_mean_map_calculation(const vips::VImage &image_padding, const int num_h, const int num_w, const int adw_size,
                                        const int adw_stride, const int img_h, const int img_w) {
    auto arma_padding = x_vips_to_arma_cube(image_padding);
    arma::cube local_mean_map(num_w, num_h, 3);
    arma::cube sub_tube;
    for (int m = 0; m < num_h; m++) {
        auto adw_top = m * adw_stride;
        auto adw_bottom = adw_top + adw_size;
        for (int n = 0; n < num_w; n++) {
            auto adw_left = n * adw_stride;
            auto adw_right = adw_left + adw_size;
            sub_tube = arma_padding.tube(adw_left, adw_top, adw_right - 1, adw_bottom - 1);
            local_mean_map.tube(n, m) = arma::mean(arma::mean(sub_tube, 0), 1);
        }
    }
    int pad_h = image_padding.height();
    int pad_w = image_padding.width();
    auto m_h_ = pad_h - (adw_size - 1);
    auto m_w_ = pad_w - (adw_size - 1);

    auto top_x = int((m_w_ - img_w) / 2);
    auto top_y = int((m_h_ - img_h) / 2);
    auto idx = vips::VImage::xyz(m_w_, m_h_);
    idx = idx / adw_stride;
    idx = idx.extract_area(top_x, top_y, img_w, img_h);

//    printf("image w, h [%d, %d, %d, %d]\n", top_x, top_y, img_w, img_h);

    auto result = x_arma_cube_to_vips(local_mean_map).mapim(idx);

    return result;
}

std::vector<vips::VImage> color_transfer_padding(const vips::VImage &img_data_source,
                                                 const vips::VImage &img_data_base,
                                                 const char *img_file_name_source,
                                                 const std::map<std::string, std::vector<long long>> &query_table_tif_geo,
                                                 const std::map<std::string, std::vector<long long>> &dict_tiff_to_base,
                                                 const std::map<std::string, std::string> &query_table_geo_tif,
                                                 int adw_size_source, int adw_stride_source, int adw_size_base, int adw_stride_base) {
    int num_h_source, num_w_source, num_h_base, num_w_base;
    // 准备base影像裁切和resize
    auto c = dict_tiff_to_base.find(img_file_name_source);
    auto cut_base_img = img_data_base.extract_area(c->second[0], c->second[1], c->second[2], c->second[3]);
    cut_base_img = cut_base_img.resize((double) img_data_source.width() / cut_base_img.width(), vips::VImage::option()->
            set("vscale", (double) img_data_source.height() / cut_base_img.height())->
            set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
//    printf("resize wh [%d, %d]\n", cut_base_img.width(), cut_base_img.height());

    int padding_top_source, padding_bottom_source, padding_left_source, padding_right_source;
    int padding_top_base, padding_bottom_base, padding_left_base, padding_right_base;

    padding_calculation(img_data_source.height(), img_data_source.width(), adw_size_source, adw_stride_source, &padding_top_source,
                        &padding_bottom_source, &padding_left_source, &padding_right_source, &num_h_source, &num_w_source);
    padding_calculation(cut_base_img.height(), cut_base_img.width(), adw_size_base, adw_stride_base, &padding_top_base, &padding_bottom_base,
                        &padding_left_base, &padding_right_base, &num_h_base, &num_w_base);

    auto a = query_table_tif_geo.find(img_file_name_source);
    auto x0 = a->second[0];
    auto y0 = a->second[1];
    auto geo_w = a->second[2];
    auto geo_h = a->second[3];

    auto key_top_left = std::to_string(x0 - geo_w) + "_" + std::to_string(y0 + geo_h);
    auto key_top = std::to_string(x0) + "_" + std::to_string(y0 + geo_h);
    auto key_top_right = std::to_string(x0 + geo_w) + "_" + std::to_string(y0 + geo_h);
    auto key_left = std::to_string(x0 - geo_w) + "_" + std::to_string(y0);
    auto key_right = std::to_string(x0 + geo_w) + "_" + std::to_string(y0);
    auto key_bottom_left = std::to_string(x0 - geo_w) + "_" + std::to_string(y0 - geo_h);
    auto key_bottom = std::to_string(x0) + "_" + std::to_string(y0 - geo_h);
    auto key_bottom_right = std::to_string(x0 + geo_w) + "_" + std::to_string(y0 - geo_h);


    vips::VImage img_top_left_source, img_top_source, img_top_right_source, img_left_source,
            img_right_source, img_bottom_left_source, img_bottom_source, img_bottom_right_source;

    vips::VImage img_top_left_base, img_top_base, img_top_right_base, img_left_base,
            img_right_base, img_bottom_left_base, img_bottom_base, img_bottom_right_base;

    vips::VImage image_padding_source = vips::VImage::black(img_data_source.width() + padding_left_source + padding_right_source,
                                                            img_data_source.height() + padding_top_source + padding_bottom_source);

    vips::VImage image_padding_base = vips::VImage::black(cut_base_img.width() + padding_left_base + padding_right_base,
                                                          cut_base_img.height() + padding_top_base + padding_bottom_base);


    auto flip_horizontal_source = img_data_source.flip(VIPS_DIRECTION_HORIZONTAL);
    auto flip_vertical_source = img_data_source.flip(VIPS_DIRECTION_VERTICAL);
    auto flip_twice_source = img_data_source.flip(VIPS_DIRECTION_HORIZONTAL);
    flip_twice_source = flip_twice_source.flip(VIPS_DIRECTION_VERTICAL);

    auto flip_horizontal_base = cut_base_img.flip(VIPS_DIRECTION_HORIZONTAL);
    auto flip_vertical_base = cut_base_img.flip(VIPS_DIRECTION_VERTICAL);
    auto flip_twice_base = cut_base_img.flip(VIPS_DIRECTION_HORIZONTAL);
    flip_twice_base = flip_twice_base.flip(VIPS_DIRECTION_VERTICAL);

    // ---------------------------上面一排-------------------------------- //
    auto end = query_table_geo_tif.end();
    // 左上角
    auto b = query_table_geo_tif.find(key_top_left);
    if (b != end) {
        img_top_left_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_top_left_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_top_left_base = img_top_left_base.resize((double) cut_base_img.width() / img_top_left_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_top_left_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_top_left_source = flip_twice_source;
        img_top_left_base = flip_twice_base;
    }
    image_padding_source = image_padding_source.insert(img_top_left_source.extract_area(img_top_left_source.width() - padding_left_source,
                                                                                        img_top_left_source.height() - padding_top_source,
                                                                                        padding_left_source, padding_top_source),
                                                       0, 0);
    image_padding_base = image_padding_base.insert(img_top_left_base.extract_area(img_top_left_base.width() - padding_left_base,
                                                                                  img_top_left_base.height() - padding_top_base,
                                                                                  padding_left_base, padding_top_base),
                                                   0, 0);

    // 上面
    b = query_table_geo_tif.find(key_top);
    if (b != end) {
        img_top_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_top_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_top_base = img_top_base.resize((double) cut_base_img.width() / img_top_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_top_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_top_source = flip_vertical_source;
        img_top_base = flip_vertical_base;
    }
    image_padding_source = image_padding_source.insert(img_top_source.extract_area(0, img_top_source.height() - padding_top_source,
                                                                                   img_top_source.width(), padding_top_source),
                                                       padding_left_source, 0);
    image_padding_base = image_padding_base.insert(img_top_base.extract_area(0, img_top_base.height() - padding_top_base,
                                                                             img_top_base.width(), padding_top_base),
                                                   padding_left_base, 0);

    // 右上角
    b = query_table_geo_tif.find(key_top_right);
    if (b != end) {
        img_top_right_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_top_right_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_top_right_base = img_top_right_base.resize((double) cut_base_img.width() / img_top_right_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_top_right_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_top_right_source = flip_twice_source;
        img_top_right_base = flip_twice_base;
    }
    image_padding_source = image_padding_source.insert(img_top_right_source.extract_area(0, img_top_right_source.height() - padding_top_source,
                                                                                         padding_right_source, padding_top_source),
                                                       padding_left_source + img_data_source.width(), 0);
    image_padding_base = image_padding_base.insert(img_top_right_base.extract_area(0, img_top_right_base.height() - padding_top_base,
                                                                                   padding_right_base, padding_top_base),
                                                   padding_left_base + cut_base_img.width(), 0);

//    x_display_vips_image((image_padding_source * 255).cast(VIPS_FORMAT_UCHAR), "source", 0);
//    x_display_vips_image((image_padding_base * 255).cast(VIPS_FORMAT_UCHAR), "base", 0);
//    cv::waitKey();
//    cv::destroyAllWindows();

    // ---------------------------中间一排-------------------------------- //
    // 左边
    b = query_table_geo_tif.find(key_left);
    if (b != end) {
        img_left_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_left_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_left_base = img_left_base.resize((double) cut_base_img.width() / img_left_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_left_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_left_source = flip_horizontal_source;
        img_left_base = flip_horizontal_base;
    }
    image_padding_source = image_padding_source.insert(img_left_source.extract_area(img_left_source.width() - padding_left_source,
                                                                                    0, padding_left_source, img_left_source.height()),
                                                       0, padding_top_source);
    image_padding_base = image_padding_base.insert(img_left_base.extract_area(img_left_base.width() - padding_left_base,
                                                                              0, padding_left_base, img_left_base.height()),
                                                   0, padding_top_base);


//    // 加入自己，居中那张
    image_padding_source = image_padding_source.insert(img_data_source, padding_left_source, padding_top_source);
    image_padding_base = image_padding_base.insert(cut_base_img, padding_left_base, padding_top_base);


//    x_display_vips_image((image_padding_source * 255).cast(VIPS_FORMAT_UCHAR), "source", 0);
//    x_display_vips_image((image_padding_base * 255).cast(VIPS_FORMAT_UCHAR), "base", 0);
//    cv::waitKey();
//    cv::destroyAllWindows();

    // 右边
    b = query_table_geo_tif.find(key_right);
    if (b != end) {
        img_right_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_right_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_right_base = img_right_base.resize((double) cut_base_img.width() / img_right_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_right_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_right_source = flip_horizontal_source;
        img_right_base = flip_horizontal_base;
    }
    image_padding_source = image_padding_source.insert(img_right_source.extract_area(0, 0, padding_right_source, img_right_source.height()),
                                                       padding_left_source + img_data_source.width(), padding_top_source);
    image_padding_base = image_padding_base.insert(img_right_base.extract_area(0, 0, padding_right_base, img_right_base.height()),
                                                   padding_left_base + cut_base_img.width(), padding_top_base);

//    x_display_vips_image((image_padding_source * 255).cast(VIPS_FORMAT_UCHAR), "source", 0);
//    x_display_vips_image((image_padding_base * 255).cast(VIPS_FORMAT_UCHAR), "base", 0);
//    cv::waitKey();
//    cv::destroyAllWindows();

    // ---------------------------下面一排-------------------------------- //

//     左下角
    b = query_table_geo_tif.find(key_bottom_left);
    if (b != end) {
        img_bottom_left_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_bottom_left_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_bottom_left_base = img_bottom_left_base.resize((double) cut_base_img.width() / img_bottom_left_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_bottom_left_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_bottom_left_source = flip_twice_source;
        img_bottom_left_base = flip_twice_base;
    }
    image_padding_source = image_padding_source.insert(img_bottom_left_source.extract_area(img_bottom_left_source.width() - padding_left_source,
                                                                                           0, padding_left_source, padding_bottom_source),
                                                       0, img_data_source.height() + padding_bottom_source);

    image_padding_base = image_padding_base.insert(img_bottom_left_base.extract_area(img_bottom_left_base.width() - padding_left_base,
                                                                                     0, padding_left_base, padding_bottom_base),
                                                   0, cut_base_img.height() + padding_bottom_base);


//    // 下面
    b = query_table_geo_tif.find(key_bottom);
    if (b != end) {
        img_bottom_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_bottom_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_bottom_base = img_bottom_base.resize((double) cut_base_img.width() / img_bottom_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_bottom_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_bottom_source = flip_vertical_source;
        img_bottom_base = flip_vertical_base;
    }
    image_padding_source = image_padding_source.insert(img_bottom_source.extract_area(0, 0, img_bottom_source.width(), padding_bottom_source),
                                                       padding_left_source, img_data_source.height() + padding_top_source);

    image_padding_base = image_padding_base.insert(img_bottom_base.extract_area(0, 0, img_bottom_base.width(), padding_bottom_base),
                                                   padding_left_base, cut_base_img.height() + padding_top_base);


//    // 右下角
    b = query_table_geo_tif.find(key_bottom_right);
    if (b != end) {
        img_bottom_right_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_bottom_right_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_bottom_right_base = img_bottom_right_base.resize((double) cut_base_img.width() / img_bottom_right_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_bottom_right_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_bottom_right_source = flip_twice_source;
        img_bottom_right_base = flip_twice_base;
    }
    image_padding_source = image_padding_source.insert(img_bottom_right_source.extract_area(0, 0, padding_right_source, padding_bottom_source),
                                                       img_data_source.width() + padding_left_source, img_data_source.height() + padding_top_source);

    image_padding_base = image_padding_base.insert(img_bottom_right_base.extract_area(0, 0, padding_right_base, padding_bottom_base),
                                                   cut_base_img.width() + padding_left_base, cut_base_img.height() + padding_top_base);

//    x_display_vips_image((image_padding_source * 255).cast(VIPS_FORMAT_UCHAR), img_file_name_source, 0);
//    x_display_vips_image((image_padding_base * 255).cast(VIPS_FORMAT_UCHAR), "base", 0);
//    cv::waitKey();
//    cv::destroyAllWindows();

    // 开始计算local mean map
    auto local_mean_source = local_mean_map_calculation(image_padding_source, num_h_source, num_w_source, adw_size_source, adw_stride_source,
                                                        img_data_source.height(), img_data_source.width());
    auto local_mean_base = local_mean_map_calculation(image_padding_base, num_h_base, num_w_base, adw_size_base, adw_stride_base,
                                                      cut_base_img.height(), cut_base_img.width());


    return {local_mean_source, local_mean_base};
//    return {};

}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_cache_set_max(10);
    vips_leak_set(TRUE);
//    system("chcp 65001");

//    const char *base_name = R"(D:\xmap_test_imagedata\seqian\base\base.tif)";
//    const char *origin_dir = R"(D:\xmap_test_imagedata\seqian\source)";

    const char *base_name = R"(D:\xmap_test_imagedata\mapping\seqian\dt2.tif)";
    const char *origin_dir = R"(D:\xmap_test_imagedata\mapping\seqian\ys)";
    const char *output_dir = R"(D:\xmap_test_imagedata\output)";

    auto base_image = vips::VImage::new_from_file(base_name);

    std::map<std::string, std::string> dict_tiffs_tfws, query_table_geo_tif;
//    std::map<std::string, std::string>::iterator it;
    std::map<std::string, std::vector<long long>> dict_tiff_to_base, query_table_tif_geo;
//    std::map<std::string, std::vector<long long>>::iterator t;

    std::filesystem::path base_path(base_name);
    auto base_tfw_name = base_path.parent_path().string() + "\\" + base_path.stem().string() + ".tfw";

    int full_h, full_w, img_num;
    get_query_tables(base_tfw_name.c_str(), origin_dir, &dict_tiffs_tfws, &query_table_geo_tif, &dict_tiff_to_base,
                     &query_table_tif_geo, &full_h, &full_w, &img_num);

//    for (it = dict_tiffs_tfws.begin(); it != dict_tiffs_tfws.end(); it++) {
//        std::cout << it->first << ", " << it->second << "\n";
//    }
//    for (it = query_table_geo_tif.begin(); it != query_table_geo_tif.end(); it++) {
//        std::cout << it->first << ", " << it->second << "\n";
//    }
//    for (t = dict_tiff_to_base.begin(); t != dict_tiff_to_base.end(); t++) {
//        auto v = t->second;
//        printf("%s, [%d, %d, %d, %d]\n", t->first.c_str(), v[0], v[1], v[2], v[3]);
//    }
//    for (t = query_table_tif_geo.begin(); t != query_table_tif_geo.end(); t++) {
//        auto v = t->second;
//        printf("%s, [%d, %d, %d, %d]\n", t->first.c_str(), v[0], v[1], v[2], v[3]);
//    }
//    return 0;

    int adw_size_base, adw_stride_base, adw_size_source, adw_stride_source;
    auto start = x_get_current_ms();
//    global_adw_stride(dict_tiff_to_base, base_image, 0.1, 15432, 23148, 0.2, 16,
//                      &adw_size_base, &adw_stride_base, &adw_size_source, &adw_stride_source);

    global_adw_stride(dict_tiff_to_base, base_image, 0.05, 0.2, full_h, full_w, img_num,
                      &adw_size_base, &adw_stride_base, &adw_size_source, &adw_stride_source);
    auto end = x_get_current_ms();
    printf("adw calculation time : %f second\n", double(end - start) / 1000);
    printf("base[%d, %d], source[%d, %d]\n", adw_size_base, adw_stride_base, adw_size_source, adw_stride_source);

//    int adw_size_base = 3251, adw_stride_base = 2600, adw_size_source = 4919, adw_stride_source = 3934;
    float alpha = 1.0;

    for (const auto &entry: std::filesystem::directory_iterator(origin_dir)) {
        auto ext = entry.path().extension();
        if (ext == ".tif") {
            auto img_data = vips::VImage::new_from_file(entry.path().string().c_str()) / 255.;
            img_data = img_data[0].bandjoin(img_data[1]).bandjoin(img_data[2]);
            auto local_mean_maps = color_transfer_padding(img_data, base_image, entry.path().string().c_str(), query_table_tif_geo, dict_tiff_to_base,
                                                          query_table_geo_tif, adw_size_source, adw_stride_source, adw_size_base, adw_stride_base);

            auto gamma = local_mean_maps[1].log() / local_mean_maps[0].log();
            auto dst = alpha * img_data.pow(gamma);
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            auto output_name = std::string(output_dir) + "\\" + entry.path().filename().string();
            dst.write_to_file(output_name.c_str());

        }
    }
    auto end1 = x_get_current_ms();
    printf("color transfer time : %f second\n", double(end1 - end) / 1000);
    printf("done process.\n");

    return 0;
}