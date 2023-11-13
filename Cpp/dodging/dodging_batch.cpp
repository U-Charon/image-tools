//
// Created by jerry on 2022/6/22.
//

#include "../../include/xmap_util.hpp"
#include "../../include/cmdline.h"
#include <filesystem>

void save(vips::VImage &input, const char *filename) {
    input.write_to_file(filename, vips::VImage::option()->
            set("compression", VIPS_FOREIGN_TIFF_COMPRESSION_NONE)->
            set("Q", 100));
}

void get_tfw_content(const char *filename, double *result) {
    std::ifstream ifs;
    std::string line;
    int i = 0;

    ifs.open(filename, std::ios::in);
    if (!ifs.is_open()) {
        printf("open tfw file %s failed.", filename);
        return;
    }
    while (getline(ifs, line)) {
        if (!line.empty()) {
            result[i] = std::stod(line);
        }
        i += 1;
    }
    ifs.close();
}

vips::VImage get_3_bands_image(const char *filename) {
    auto img = vips::VImage::new_from_file(filename) / 255;
    if (img.bands() == 4) {
        return img[0].bandjoin(img[1]).bandjoin(img[2]);
    } else {
        return img;
    }
}

void global_adw(const char *directory, int *adw_size, int *adw_stride,
                std::map<std::string, std::string> *dict_query_tif,
                std::map<std::string, std::vector<int>> *dict_tif_properties,
                float p, int full_h, int full_w, float overlap = 0.2, int img_num = 9) {
    using namespace arma;
    vips::VImage o_img_data;
    fmat images_stat(img_num, 3, fill::zeros);
    int i = 0;
    double tfw_result[6];

    for (const auto &entry: std::filesystem::directory_iterator(directory)) {
        int img_h, img_w;

        auto filename = entry.path().string();
        size_t pos = filename.find('.', 0);
        auto suffix = filename.substr(pos + 1, 3);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);

        if (suffix == "tif") {
            o_img_data = vips::VImage::new_from_file(filename.c_str()) / 255;
            images_stat(i, 0) = (float) o_img_data.avg();
            auto temp_sum = (o_img_data.avg() - o_img_data).pow(2.0);
            auto img_sum = x_vips_to_arma_cube(temp_sum);
            fmat a = sum(conv_to<fcube>::from(img_sum), 2);
            images_stat(i, 1) = sum(sum(a)); // 组内方差
            images_stat(i, 2) = (float) o_img_data.width() * (float) o_img_data.height() * 3;
            i += 1;

            img_h = o_img_data.height();
            img_w = o_img_data.width();

            // process tif->tfw file
            auto tfw_file_name = filename;
            tfw_file_name = tfw_file_name.replace(tfw_file_name.find("tif"), 3, "tfw");
            get_tfw_content(tfw_file_name.c_str(), &tfw_result[0]);
            // 放大10倍确保能跟h，w凑上
            int scale = 10;
//            auto x0 = int((tfw_result[4] + tfw_result[0] / 2) * scale);
//            auto y0 = int((tfw_result[5] + tfw_result[3] / 2) * scale);
//            auto xn = int((tfw_result[4] + tfw_result[0] / 2 + tfw_result[0] * img_w) * scale);
//            auto yn = int((tfw_result[5] + tfw_result[3] / 2 + tfw_result[3] * img_h) * scale);
            auto x0 = int((tfw_result[4] - tfw_result[0] / 2) * scale);
            auto y0 = int((tfw_result[5] + tfw_result[0] / 2) * scale);
            auto xn = int((tfw_result[4] - tfw_result[0] / 2 + tfw_result[0] * img_w) * scale);
            auto yn = int((tfw_result[5] + tfw_result[0] / 2 - tfw_result[0] * img_h) * scale);


            auto geo_h = y0 - yn;
            auto geo_w = xn - x0;
            auto x0_y0 = std::to_string(x0) + "_" + std::to_string(y0);
//            printf("[%s] = %s\n", x0_y0.c_str(), tif_file_name.c_str());
            dict_query_tif->insert(std::map<std::string, std::string>::value_type(x0_y0, filename));
            std::vector<int> v = {x0, y0, geo_w, geo_h};
            dict_tif_properties->insert(std::map<std::string, std::vector<int>>::value_type(filename, v));

        }
//        if (suffix == "tfw") {
////            printf("tfw file... %s\n", filename.c_str());
//            get_tfw_content(filename.c_str(), &tfw_result[0]);
////            printf("%s: [%f, %f, %f, %f, %f, %f]\n", filename.c_str(), tfw_result[0], tfw_result[1], tfw_result[2], tfw_result[3], tfw_result[4], tfw_result[5]);
//
//            // 放大10倍确保能跟h，w凑上
//            int scale = 10;
//            auto x0 = int((tfw_result[4] + tfw_result[0] / 2) * scale);
//            auto y0 = int((tfw_result[5] + tfw_result[3] / 2) * scale);
//            auto xn = int((tfw_result[4] + tfw_result[0] / 2 + tfw_result[0] * img_w) * scale);
//            auto yn = int((tfw_result[5] + tfw_result[3] / 2 + tfw_result[3] * img_h) * scale);
//            auto geo_h = y0 - yn;
//            auto geo_w = xn - x0;
//
//            auto x0_y0 = std::to_string(x0) + "_" + std::to_string(y0);
//            auto tif_file_name = filename.replace(filename.find("tfw"), 3, "tif");
////            printf("[%s] = %s\n", x0_y0.c_str(), tif_file_name.c_str());
//            dict_query_tif->insert(std::map<std::string, std::string>::value_type(x0_y0, tif_file_name));
//            std::vector<int> a = {x0, y0, geo_w, geo_h};
//            dict_tif_properties->insert(std::map<std::string, std::vector<int>>::value_type(tif_file_name, a));
//        }
    } // end for loop

    auto images_mean = sum(images_stat.col(0) % images_stat.col(2)) / sum(images_stat.col(2));
    auto images_SSA = sum(pow((images_stat.col(0) - images_mean), 2.0) % images_stat.col(2)); // 组间方差
    auto images_SSE = sum(images_stat.col(1)); // 组内方差
    auto images_SST = images_SSA + images_SSE; // 总方差
    auto images_std = sqrt(images_SST / sum(images_stat.col(2)));
//    printf("%f, %f, %f, %f, %f\n", images_mean, images_SSA, images_SSE, images_SST, images_std);
    float constant = 128.0 / 45;  //理想状态下的mean/std
    auto rho = p / images_std * images_mean / constant;  // std越大，ρ越小， ADWs的size小
    *adw_size = int(sqrt(rho * full_h * rho * full_w) / 2) * 2 + 1;  // 奇数
    auto temp = *adw_size;
    *adw_stride = int(temp * (1 - overlap) / 2) * 2;  // 偶数

}

vips::VImage calculate_local_mean_map_batch(const vips::VImage &img_data,
                                            const char *img_file_name,
                                            std::map<std::string, std::string> *dict_query_tif,
                                            std::map<std::string, std::vector<int>> *dict_tif_properties,
                                            int adw_size, int adw_stride
) {

    auto h = img_data.height();
    auto w = img_data.width();

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

    auto a = dict_tif_properties->find(img_file_name);
    if (a == dict_tif_properties->end()) {
        printf("Can't find tif file name.");
        return nullptr;
    }
//    printf("%s = [%d, %d, %d, %d]\n", img_file_name, a->second[0], a->second[1], a->second[2], a->second[3]);
//    printf("padding: [%d, %d, %d, %d]\n", padding_top, padding_bottom, padding_left, padding_right);
//    printf("num_h=%d, num_w=%d\n", num_h, num_w);
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

//    printf("%s, %s, %s, %s, %s, %s, %s, %s\n", key_top_left.c_str(), key_top.c_str(), key_top_right.c_str(), key_left.c_str(),
//           key_right.c_str(), key_bottom_left.c_str(), key_bottom.c_str(), key_bottom_right.c_str());

    vips::VImage img_top_left, img_top, img_top_right, img_left,
            img_right, img_bottom_left, img_bottom, img_bottom_right;

    vips::VImage image_padding = vips::VImage::black(img_data.width() + padding_left + padding_right,
                                                     img_data.height() + padding_top + padding_bottom);

    auto flip_horizontal = img_data.flip(VIPS_DIRECTION_HORIZONTAL);
    auto flip_vertical = img_data.flip(VIPS_DIRECTION_VERTICAL);
    auto flip_twice = img_data.flip(VIPS_DIRECTION_HORIZONTAL);
    flip_twice = flip_twice.flip(VIPS_DIRECTION_VERTICAL);

    // ---------------------------上面一排-------------------------------- //
    auto end = dict_query_tif->end();
    // 左上角
    auto b = dict_query_tif->find(key_top_left);
    if (b != end) {
//        img_top_left = vips::VImage::new_from_file(b->second.c_str()) / 255;
        img_top_left = get_3_bands_image(b->second.c_str());
    } else {
        img_top_left = flip_twice;
    }
    image_padding = image_padding.insert(img_top_left.extract_area(img_top_left.width() - padding_left,
                                                                   img_top_left.height() - padding_top,
                                                                   padding_left, padding_top),
                                         0, 0);

    // 上面
    b = dict_query_tif->find(key_top);
    if (b != end) {
        img_top = get_3_bands_image(b->second.c_str());
    } else {
        img_top = flip_vertical;
    }
    image_padding = image_padding.insert(img_top.extract_area(0, img_top.height() - padding_top,
                                                              img_top.width(), padding_top),
                                         padding_left, 0);

    // 右上角
    b = dict_query_tif->find(key_top_right);
    if (b != end) {
        img_top_right = get_3_bands_image(b->second.c_str());
    } else {
        img_top_right = flip_twice;
    }
    image_padding = image_padding.insert(img_top_right.extract_area(0, img_top_right.height() - padding_top,
                                                                    padding_right, padding_top),
                                         padding_left + img_data.width(), 0);

    // ---------------------------中间一排-------------------------------- //
    // 左边
    b = dict_query_tif->find(key_left);
    if (b != end) {
        img_left = get_3_bands_image(b->second.c_str());
    } else {
        img_left = flip_horizontal;
    }
    image_padding = image_padding.insert(img_left.extract_area(img_left.width() - padding_left,
                                                               0,
                                                               padding_left, img_left.height()),
                                         0, padding_top);

    // 加入自己，居中那张
    image_padding = image_padding.insert(img_data, padding_left, padding_top);

    // 右边
    b = dict_query_tif->find(key_right);
    if (b != end) {
        img_right = get_3_bands_image(b->second.c_str());
    } else {
        img_right = flip_horizontal;
    }
    image_padding = image_padding.insert(img_right.extract_area(0,
                                                                0,
                                                                padding_right, img_right.height()),
                                         padding_left + img_data.width(), padding_top);

    // ---------------------------下面一排-------------------------------- //

    // 左下角
    b = dict_query_tif->find(key_bottom_left);
    if (b != end) {
        img_bottom_left = get_3_bands_image(b->second.c_str());
    } else {
        img_bottom_left = flip_twice;
    }
    image_padding = image_padding.insert(img_bottom_left.extract_area(img_bottom_left.width() - padding_left,
                                                                      0,
                                                                      padding_left, padding_bottom),
                                         0, img_data.height() + padding_bottom);


    // 下面
    b = dict_query_tif->find(key_bottom);
    if (b != end) {
        img_bottom = get_3_bands_image(b->second.c_str());
    } else {
        img_bottom = flip_vertical;
    }
    image_padding = image_padding.insert(img_bottom.extract_area(0, 0,
                                                                 img_bottom.width(), padding_bottom),
                                         padding_left, img_data.height() + padding_top);

    // 右下角
    b = dict_query_tif->find(key_bottom_right);
    if (b != end) {
        img_bottom_right = get_3_bands_image(b->second.c_str());
    } else {
        img_bottom_right = flip_twice;
    }
    image_padding = image_padding.insert(img_bottom_right.extract_area(0,
                                                                       0,
                                                                       padding_right, padding_bottom),
                                         img_data.width() + padding_left, img_data.height() + padding_top);

//    x_display_vips_image((image_padding * 255).cast(VIPS_FORMAT_UCHAR), "pad", 0);
//    x_display_vips_image((img_data * 255).cast(VIPS_FORMAT_UCHAR), "data", 1);
//    cv::waitKey();
//    cv::destroyAllWindows();
//    auto o = (image_padding * 255).cast(VIPS_FORMAT_UCHAR);
//    save(o, R"(D:\test_stretch_images\trans1\target\A2.tif)");
//    save(o, R"(D:\test_stretch_images\trans1\target\B2.tif)");

    // 开始计算local mean map
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

    auto top_x = int((m_w_ - w) / 2);
    auto top_y = int((m_h_ - h) / 2);
    auto idx = vips::VImage::xyz(m_w_, m_h_);
    idx = idx / adw_stride;
    idx = idx.extract_area(top_x, top_y, w, h);

    auto cv_lmm = x_arma_to_cv(local_mean_map);
    auto result = x_cv_to_vips_double(cv_lmm).mapim(idx);

//    x_display_vips_image((result * 255).cast(VIPS_FORMAT_UCHAR), "data", 1);
//    cv::waitKey();
//    cv::destroyAllWindows();

    return result;

//    auto end = dict_query_tif->end();
//    // 左上角
//    auto b = dict_query_tif->find(key_top_left);
//    if (b != end) {
//        printf("[%s] has top left picture: [%s]\n", a->first.c_str(), b->second.c_str());
//    } else {
//        printf("key:%s, [%s] no top left\n", key_top_left.c_str(), a->first.c_str());
//    }
//    // 上面
//    b = dict_query_tif->find(key_top);
//    if (b != end) {
//        printf("[%s] has top picture: [%s]\n", a->first.c_str(), b->second.c_str());
//    } else {
//        printf("key:%s, [%s] no top\n", key_top_left.c_str(), a->first.c_str());
//    }
//    // 右上角
//    b = dict_query_tif->find(key_top_right);
//    if (b != end) {
//        printf("[%s] has top right picture: [%s]\n", a->first.c_str(), b->second.c_str());
//    } else {
//        printf("key:%s, [%s] no top right\n", key_top_left.c_str(), a->first.c_str());
//    }
//    // 左边
//    b = dict_query_tif->find(key_left);
//    if (b != end) {
//        printf("[%s] has left picture: [%s]\n", a->first.c_str(), b->second.c_str());
//    } else {
//        printf("key:%s, [%s] no left\n", key_top_left.c_str(), a->first.c_str());
//    }
//    // 右边
//    b = dict_query_tif->find(key_right);
//    if (b != end) {
//        printf("[%s] has right picture: [%s]\n", a->first.c_str(), b->second.c_str());
//    } else {
//        printf("key:%s, [%s] no right\n", key_top_left.c_str(), a->first.c_str());
//    }
//    // 左下角
//    b = dict_query_tif->find(key_bottom_left);
//    if (b != end) {
//        printf("[%s] has bottom left picture: [%s]\n", a->first.c_str(), b->second.c_str());
//    } else {
//        printf("key:%s, [%s] no bottom left picture\n", key_top_left.c_str(), a->first.c_str());
//    }
//    // 下面
//    b = dict_query_tif->find(key_bottom);
//    if (b != end) {
//        printf("[%s] has bottom picture: [%s]\n", a->first.c_str(), b->second.c_str());
//    } else {
//        printf("key:%s, [%s] no bottom picture\n", key_top_left.c_str(), a->first.c_str());
//    }
//    // 右下角
//    b = dict_query_tif->find(key_bottom_right);
//    if (b != end) {
//        printf("[%s] has bottom right picture: [%s]\n", a->first.c_str(), b->second.c_str());
//    } else {
//        printf("key:%s, [%s] no bottom right picture\n", key_top_left.c_str(), a->first.c_str());
//    }
//
//    return nullptr;
}

void copy_tfw_file(const char *in, const char *out) {
    std::fstream fin, fout;
    fin.open(in, std::ios_base::in);
    fout.open(out, std::ios_base::out | std::ios_base::trunc);
    if (!fin.is_open() || !fout.is_open()) {
        printf("open twf file error");
        exit(0);
    }

    char *buf = (char *) calloc(1, 1024);
    while (!fin.eof()) {
        fin.read(buf, 1024);
        fout.write(buf, fin.gcount());
    }
    //关闭文件，释放内存
    fin.close();
    fout.close();
    free(buf);
}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_cache_set_max(10);
    vips_leak_set(TRUE);

    //    int h = 4892 + 4892;
//    int w = 4892 + 4892;
//    const char *origin_dir = "/Users/Jerry/dev/20220401-balancetest/batch/trans/target";
//    const char *source_dir = "/Users/Jerry/dev/20220401-balancetest/batch/trans/source";

//    int h = 271 + 271 + 269;
//    int w = 344 + 344 + 343;
//    const char *origin_dir = R"(D:\test_stretch_images\trans\target)";
//    const char *source_dir = R"(D:\test_stretch_images\trans\source)";

    int h = 5000 * 3;
    int w = 5000 * 3;
    float alpha = 1.0;
    float overlap = 0.2;
    float p = 0.1;
    int img_nums = 9;
    const char *origin_dir = "/Users/Jerry/dev/20220401-balancetest/batch/trans1/target";
    const char *source_dir = "/Users/Jerry/dev/20220401-balancetest/batch/trans1/source";
    const char *output_dir = "/Users/Jerry/dev/20220401-balancetest/batch/trans1/output";

// dodging_batch.exe --base="D:\test_stretch_images\trans1\source" --raw="D:\test_stretch_images\trans1\target" --out="D:\test_stretch_images\trans1\output" --p=0.1 --overlap=0.2 --alpha=1.0 --height=15000 --width=15000 --img_nums=9
//    cmdline::parser theArgs; // from GO: sourceDir, inputDir, outputDir string, p, overlap, alpha float64, h, w int
//    theArgs.add<std::string>("base", '\0', "base picture directory", true);
//    theArgs.add<std::string>("raw", '\0', "raw picture directory", true);
//    theArgs.add<std::string>("out", '\0', "output picture directory", true);
//    theArgs.add<float>("p", '\0', "p parameter", true);
//    theArgs.add<float>("overlap", '\0', "overlap parameter", true);
//    theArgs.add<float>("alpha", '\0', "alpha parameter", true);
//    theArgs.add<int>("height", '\0', "total image height", true);
//    theArgs.add<int>("width", '\0', "total image width", true);
//    theArgs.add<int>("img_nums", '\0', "total image numbers", true);
//
//    theArgs.parse_check(argc, argv);
//
//    int h = theArgs.get<int>("height");
//    int w = theArgs.get<int>("width");
//    float alpha = theArgs.get<float>("alpha");
//    float overlap = theArgs.get<float>("overlap");
//    float p = theArgs.get<float>("p");
//    int img_nums = theArgs.get<int>("img_nums");
//    const char *origin_dir = theArgs.get<std::string>("raw").c_str();
//    const char *source_dir = theArgs.get<std::string>("base").c_str();
//    const char *output_dir = theArgs.get<std::string>("out").c_str();


    int a, b, c, d;
    std::map<std::string, std::string> dict_query_tif_s;
    std::map<std::string, std::string> dict_query_tif_o;
    std::map<std::string, std::vector<int>> dict_tif_properties_s;
    std::map<std::string, std::vector<int>> dict_tif_properties_o;


    // 1. compute adw
    global_adw(origin_dir, &c, &d, &dict_query_tif_o,
               &dict_tif_properties_o, p, h, w, overlap, img_nums);
    global_adw(source_dir, &a, &b, &dict_query_tif_s,
               &dict_tif_properties_s, p, h, w, overlap, img_nums);

//    auto it0 = dict_query_tif_o.begin();
//    while (it0 != dict_query_tif_o.end()) {
//        printf("[%s] = %s] \n", it0->first.c_str(), it0->second.c_str());
//        it0++;
//    }
//    auto it1 = dict_tif_properties_o.begin();
//    while (it1 != dict_tif_properties_o.end()) {
//        auto value = it1->second;
//        printf("tif [%s] properties: [%d, %d, %d, %d] \n", it1->first.c_str(), value[0], value[1], value[2], value[3]);
//        it1++;
//    }



    // 2. color transfer
//    auto img_data = vips::VImage::new_from_file(R"(D:\test_stretch_images\trans1\target\target_A2.tif)") / 255;
//    img_data = img_data[0].bandjoin(img_data[1]).bandjoin(img_data[2]);
//    auto o_mean_map = calculate_local_mean_map_batch(img_data, R"(D:\test_stretch_images\trans1\target\target_A2.tif)",
//                                                     &dict_query_tif_o, &dict_tif_properties_o, c, d);
//
//    return 0;

    for (const auto &entry: std::filesystem::directory_iterator(origin_dir)) {
        vips::VImage o_img_data, s_img_data, band4;
        int bands;

        auto filename = entry.path().string();
        size_t pos = filename.find('.', 0);
        auto suffix = filename.substr(pos + 1, 3);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);

        if (suffix == "tif") {
            printf("processing: \t%s\n", filename.c_str());
            o_img_data = vips::VImage::new_from_file(filename.c_str()) / 255;
            bands = o_img_data.bands();
            if (bands == 4) {
                band4 = o_img_data[3];
                o_img_data = o_img_data[0].bandjoin(o_img_data[1]).bandjoin(o_img_data[2]);
            }
            auto o_mean_map = calculate_local_mean_map_batch(o_img_data, filename.c_str(), &dict_query_tif_o,
                                                             &dict_tif_properties_o, c, d);

            auto source_name = filename;
            auto tif_file_name = source_name.substr(source_name.rfind('/') + 1, source_name.size() - std::string(origin_dir).size());
            source_name = std::string(source_dir).append("/").append(tif_file_name);
            s_img_data = vips::VImage::new_from_file(source_name.c_str()) / 255;
            auto s_mean_map = calculate_local_mean_map_batch(s_img_data, source_name.c_str(), &dict_query_tif_s,
                                                             &dict_tif_properties_s, a, b);

//            auto img = vips::VImage::new_from_file(filename.c_str()) / 255;
            auto gamma = s_mean_map.log() / o_mean_map.log();
            auto dst = alpha * o_img_data.pow(gamma);
            if (bands == 4) {
                dst = dst.bandjoin(band4);
            }
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

//            auto output_name = filename;
//            output_name = output_name.replace(output_name.find("target"), 6, "output");
//            save(dst, output_name.c_str());
            auto output_name = std::string(output_dir).append("/").append(tif_file_name);;
            save(dst, output_name.c_str());

            auto tfw_file_in = filename;
            auto tfw_file_out = output_name;
            tfw_file_in = tfw_file_in.replace(tfw_file_in.find("tif"), 3, "tfw");
            tfw_file_out = tfw_file_out.replace(tfw_file_out.find("tif"), 3, "tfw");
            copy_tfw_file(tfw_file_in.c_str(), tfw_file_out.c_str());

        }
    }

//    for (const auto &entry: std::filesystem::directory_iterator(origin_dir)) {
//        vips::VImage o_img_data, s_img_data, band4;
//        int bands;
//
//        auto filename = entry.path().string();
//        size_t pos = filename.find('.', 0);
//        auto suffix = filename.substr(pos + 1, 3);
//        std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
//
//        if (suffix == "tif") {
//            printf("filename: %s - \t", filename.c_str());
//            auto source_name = filename;
//            source_name = source_name.substr(source_name.rfind('\\') + 1, source_name.size() - std::string(origin_dir).size());
//            source_name = std::string(source_dir).append("\\").append(source_name);
//            printf("source name: %s\n", source_name.c_str());
//        }
//    }






//    global_adw(source_dir, &c, &d, 0.05, h, w);

//    printf("origin [%d, %d]\n", a, b);
//    auto it = dict_tif_properties.begin();
//    auto it0 = dict_query_tif.begin();
//    while (it != dict_tif_properties.end()) {
//        auto value = it->second;
//        printf("tif [%s] properties: [%d, %d, %d, %d] \n", it->first.c_str(), value[0], value[1], value[2], value[3]);
//        it++;
//    }
//
//    while (it0 != dict_query_tif.end()) {
////        auto value = it0->second;
//        printf("[%s] = %s] \n", it0->first.c_str(), it0->second.c_str());
//        it0++;
//    }


//    printf("source [%d, %d]\n", c, d);

//    vips_shutdown();
    return 0;
}