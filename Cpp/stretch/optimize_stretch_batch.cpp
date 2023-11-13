//
// Created by jerry on 2022/8/15.
//
#include "../../include/xmap_util.hpp"
#include "../../include/cmdline.h"
#include <filesystem>

int find_low(const vips::VImage &in, double T) {
    auto hist = in.hist_find();

    double min_total = 0;
    int i = 1;
    while (min_total <= T) {
        min_total += hist(i, 0)[0];
        i += 1;
    }
    auto low = i - 1;

    return low;
}

int find_high(const vips::VImage &in, double T) {
    auto hist = in.hist_find();
    auto w = hist.width();
    int j;

    if (w == 65536) {
        j = w - 2;
    } else { j = w - 1; }   // 如果hist的宽度是65536，说明有白色要去掉，w-1就是65535的个数，w-2就是到65534，相当于白色剔除掉；

    double max_total = 0;
    while (max_total <= T) {
        max_total += hist(j, 0)[0];
        j -= 1;
    }
    auto high = j + 1;

    return high;
}

void calculate_hist_total(const char *directory, int bands, int order, double min_p, double max_p, int *a, int *b) {
    vips::VImage img;
    int64 pixel_number = 0;
    vips::VImage hist_total[bands];
    double min_total, max_total, T1, T2;
    int i, j;


    img = vips::VImage::black(10, 10);
    for (int k = 0; k < bands; k++) {
        hist_total[k] = img.hist_find() * 0;
    }

    // 计算所有的直方图累加
    for (const auto &entry: std::filesystem::directory_iterator(directory)) {
        auto filename = entry.path().string();
        size_t pos = filename.find('.', 0);
        auto suffix = filename.substr(pos + 1, 3);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
        if (suffix == "tif") {
            img = vips::VImage::new_from_file(filename.c_str());
            if (order == 1) {
                if (bands == 3) {
                    img = img[2].bandjoin(img[1]).bandjoin(img[0]);
                } else if (bands == 4) {
                    img = img[2].bandjoin(img[1]).bandjoin(img[0]).bandjoin(img[3]);
                }
            }

            for (int k = 0; k < bands; k++) {
                if (bands == 1) {
                    hist_total[k] += img.hist_find();
                } else {
                    hist_total[k] += img[k].hist_find();
                }
            }
            /* 2022-07-22：反馈bug，这里统计像素的时候应该排除掉0像素
             * 添加一个函数count_non_zero，统计3波段影像非零像素个数
             * 未来有可能还会去掉65535像素值
            */
//            pixel_number += img.width() * img.height();
            /*
             * 2022-07-29：在济南讨论，决定把所有非零和非65535的像素都排除掉
             */
            pixel_number += x_count_non_zero_and_white(img);
        } else if (suffix == "jpg") {
            img = vips::VImage::new_from_file(filename.c_str());
            if (order == 1) {
                if (bands == 3) {
                    img = img[2].bandjoin(img[1]).bandjoin(img[0]);
                } else if (bands == 4) {
                    img = img[2].bandjoin(img[1]).bandjoin(img[0]).bandjoin(img[3]);
                }
            }

            for (int k = 0; k < bands; k++) {
                if (bands == 1) {
                    hist_total[k] += img.hist_find();
                } else {
                    hist_total[k] += img[k].hist_find();
                }
            }

            pixel_number += x_count_non_zero_and_white(img);
        }
    }

    // 找到边界的a、b值
    T1 = min_p * pixel_number;
    T2 = (1 - max_p) * pixel_number;

    for (int m = 0; m < bands; m++) {
        min_total = 0.0;
        max_total = 0.0;
        i = 1; // i从1开始，是因为要排除掉0值；也就是说0的像素多少跟后面计算无关

        if (bands == 1) {
            // --- find a:
            while (min_total <= T1) {
                min_total += hist_total[m](i, 0)[0];
                i += 1;
            }
            a[m] = i - 1; // a值
            // --- find b:
            auto w = hist_total[m].width();
            if (w == 65536) {
                j = w - 2;
            } else { j = w - 1; }   // 如果hist的宽度是65536，说明有白色要去掉，w-1就是65535的个数，w-2就是到65534，相当于白色剔除掉；

            while (max_total <= T2) {
                max_total += hist_total[m](j, 0)[0];
                j -= 1;
            }
            b[m] = j + 1; // b值
        } else {
            // --- find a:
            while (min_total <= T1) {
                min_total += hist_total[m](i, 0)[0];
                i += 1;
            }
            a[m] = i - 1; // a值
            // --- find b:
            auto w = hist_total[m].width();
            if (w == 65536) {
                j = w - 2;
            } else { j = w - 1; }   // 如果hist的宽度是65536，说明有白色要去掉，w-1就是65535的个数，w-2就是到65534，相当于白色剔除掉；

            while (max_total <= T2) {
                max_total += hist_total[m](j, 0)[0];
                j -= 1;
            }
            b[m] = j + 1; // b值
        }
    }
}

vips::VImage optimized_stretch(const vips::VImage &in, double min_percent, double max_percent,
                               double min_adjust, double max_adjust, int order) {
//    printf("entering optimize stretch...\n");
//    auto start = x_get_current_ms();
//    VipsImage *output;
    vips::VImage input, band, dst;

    int channels = in.bands();
    vips::VImage result[channels];
    if (order == 1) {
        if (channels == 3) {
            input = in[2].bandjoin(in[1]).bandjoin(in[0]);
        } else if (channels == 4) {
            input = in[2].bandjoin(in[1]).bandjoin(in[0]).bandjoin(in[3]);
        }
    }

    double c, d, min_value = 1 / 255., max_value = 1.0;
    auto input_copy = in.copy();
    auto input_zero = (input_copy == 0).ifthenelse(0, 1);
    auto total_non_zero_pixel = x_count_non_zero_and_white(in);

    if (channels == 1) {

        auto a = find_low(in, min_percent * total_non_zero_pixel);
        auto b = find_high(in, (1 - max_percent) * total_non_zero_pixel);
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        input = (input < c).ifthenelse(min_value, input);
        auto v1 = (input - c) / (d - c) * (max_value - min_value) + min_value;
        input = (input >= c & in <= d).ifthenelse(v1, input);

        dst = (input > d).ifthenelse(max_value, input);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

    } else {
        for (int i = 0; i < channels; i++) {
            band = in.extract_band(i);
            auto a = find_low(band, min_percent * total_non_zero_pixel);
            auto b = find_high(band, (1 - max_percent) * total_non_zero_pixel);
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min_value, band);
            auto v1 = (band - c) / (d - c) * (max_value - min_value) + min_value;
            band = ((band >= c) & (band <= d)).ifthenelse(v1, band);
            result[i] = (band > d).ifthenelse(max_value, band);
        }
        if (channels == 3) {
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

        } else {
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

        }
    }

    return dst;
//    auto end = x_get_current_ms();
//    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
//    printf("done. optimize stretch...\n");
}

vips::VImage optimized_stretch_batch(const vips::VImage &in, double min_adjust, double max_adjust, int order, int channels, const int *low,
                                     const int *high) {
//    printf("entering optimize stretch batch process...\n");
//    auto start = x_get_current_ms();
//    VipsImage *output;
    vips::VImage input, band, dst;
    vips::VImage result[channels];

    if (order == 1) {
        if (channels == 3) {
            input = in[2].bandjoin(in[1]).bandjoin(in[0]);
        } else if (channels == 4) {
            input = in[2].bandjoin(in[1]).bandjoin(in[0]).bandjoin(in[3]);
        }
    } else {
        input = in;
    }

//    double a, b, c, d, min_value = 1 / 255., max_value = 254 / 255.;
    double a, b, c, d, min_value = 1 / 255., max_value = 1.0;
//    auto input_copy = in.copy();
//    auto input_zero = (input_copy == 0).ifthenelse(0, 1);
    auto input_zero = (in == 0).ifthenelse(0, 1);

    if (channels == 1) {
        a = low[0];
        b = high[0];
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        input = (input < c).ifthenelse(min_value, input);
        auto v1 = (input - c) / (d - c) * (max_value - min_value) + min_value;
        input = (input >= c & input <= d).ifthenelse(v1, input);

        dst = (input > d).ifthenelse(max_value, input);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
    } else {
        for (int i = 0; i < channels; i++) {
            band = input.extract_band(i);
            a = low[i];
            b = high[i];
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min_value, band);
            auto v1 = (band - c) / (d - c) * (max_value - min_value) + min_value;
            band = ((band >= c) & (band <= d)).ifthenelse(v1, band);
            result[i] = (band > d).ifthenelse(max_value, band);
        }
        if (channels == 3) {
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

        } else {
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

        }
    }

//    switch (channels) {
//        case 3:
//            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
//            dst = dst * input_zero;
//            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
//            break;
//        case 4:
//            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
//            dst = dst * input_zero;
//            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
//            break;
//        default:
//            break;
//    }

    return dst;
//    auto end = x_get_current_ms();
//    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
//    printf("done. optimize batch stretch...\n");
}

void save(vips::VImage &input, const char *filename) {
    input.write_to_file(filename, vips::VImage::option()->
//            set("compression", VIPS_FOREIGN_TIFF_COMPRESSION_JPEG)->
            set("Q", 75));
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
//    vips_leak_set(TRUE);

    /////////// setup parameters /////////////
    // os_batch.exe--input="D:\test_stretch_images\20220401-ImageProcess\sat\2" --output="D:\test_stretch_images\output" --min_percent=0.025 -- max_percent=0.99 --min_adjust=0.3 --max_adjust=0.3 --order=0 --channels=3
    cmdline::parser theArgs; //double min_adjust, double max_adjust, int order, int channels, const int *low, const int *high
    theArgs.add<std::string>("input", '\0', "[String]input image directory", true);
    theArgs.add<std::string>("output", '\0', "[String]output image directory", true);
    theArgs.add<float>("min_adjust", '\0', "[Float]min adjustment, left value", true);
    theArgs.add<float>("max_adjust", '\0', "[Float] adjustment, right value", true);
    theArgs.add<float>("min_percent", '\0', "[Float]min adjustment, left value", true);
    theArgs.add<float>("max_percent", '\0', "[Float] adjustment, right value", true);
    theArgs.add<int>("order", '\0', "[Int]band order(0 or 1; 0--RGB, 1--BGR)", true);
    theArgs.add<int>("channels", '\0', "[Int]image channels(1 or 3 or 4)", true);
    theArgs.add<int>("statistics", '\0', "[Bool]need statistics or not(0:No, 1:Yes)", true);

    theArgs.parse_check(argc, argv);

    const char *input_dir = theArgs.get<std::string>("input").c_str();
    const char *output_dir = theArgs.get<std::string>("output").c_str();
    float minPercent = theArgs.get<float>("min_percent");
    float maxPercent = theArgs.get<float>("max_percent");
    float minAdjustment = theArgs.get<float>("min_adjust");
    float maxAdjustment = theArgs.get<float>("max_adjust");
    int order = theArgs.get<int>("order");
    int channels = theArgs.get<int>("channels");
    bool statistics;
    int s = theArgs.get<int>("statistics");
    if (s == 0) {
        statistics = false;
    } else { statistics = true; }
    int high[channels], low[channels];

    if (statistics) {
        printf(">>>>> need statistics, calculating...\n");
        calculate_hist_total(input_dir, channels, order, minPercent,
                             maxPercent, &low[0], &high[0]);
        printf(">>>>> done statistics.\n");
        for (const auto &entry: std::filesystem::directory_iterator(input_dir)) {
            auto filename = entry.path().string(); // 文件名带目录
            size_t pos = filename.find('.', 0);
            auto suffix = filename.substr(pos + 1, 3);
            std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);

            if ((suffix == "tif") || (suffix == "jpg")) {
                printf(">>>>> processing: \t%s\n", filename.c_str());
                auto img_data = vips::VImage::new_from_file(filename.c_str());
                auto stretched = optimized_stretch_batch(img_data, minAdjustment, maxAdjustment, order, channels, low, high);
                auto output_name = std::string(output_dir).append("\\").append(entry.path().filename().string());
                save(stretched, output_name.c_str());

                auto tfw_file_in = filename;
                auto tfw_file_out = output_name;
                if (suffix == "tif") {
                    tfw_file_in = tfw_file_in.replace(tfw_file_in.find("tif"), 3, "tfw");
                    tfw_file_out = tfw_file_out.replace(tfw_file_out.find("tif"), 3, "tfw");
                } else {
                    tfw_file_in = tfw_file_in.replace(tfw_file_in.find("jpg"), 3, "tfw");
                    tfw_file_out = tfw_file_out.replace(tfw_file_out.find("jpg"), 3, "tfw");
                }

                if (std::filesystem::exists(std::filesystem::path(tfw_file_in))) {
                    copy_tfw_file(tfw_file_in.c_str(), tfw_file_out.c_str());
                }
            }
        }
    } else {
        // 不统计
        for (const auto &entry: std::filesystem::directory_iterator(input_dir)) {
            auto filename = entry.path().string();
            size_t pos = filename.find('.', 0);
            auto suffix = filename.substr(pos + 1, 3);
            std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);

            if ((suffix == "tif") || (suffix == "jpg")) {
                printf(">>>>> processing: \t%s\n", filename.c_str());
                auto img_data = vips::VImage::new_from_file(filename.c_str());
                auto stretched = optimized_stretch(img_data, minPercent, maxPercent, minAdjustment, maxAdjustment, order);
                auto output_name = std::string(output_dir).append("\\").append(entry.path().filename().string());
                save(stretched, output_name.c_str());

                auto tfw_file_in = filename;
                auto tfw_file_out = output_name;
                if (suffix == "tif") {
                    tfw_file_in = tfw_file_in.replace(tfw_file_in.find("tif"), 3, "tfw");
                    tfw_file_out = tfw_file_out.replace(tfw_file_out.find("tif"), 3, "tfw");
                } else {
                    tfw_file_in = tfw_file_in.replace(tfw_file_in.find("jpg"), 3, "tfw");
                    tfw_file_out = tfw_file_out.replace(tfw_file_out.find("jpg"), 3, "tfw");
                }
                if (std::filesystem::exists(std::filesystem::path(tfw_file_in))) {
                    copy_tfw_file(tfw_file_in.c_str(), tfw_file_out.c_str());
                }
            }
        }
    }


    return 0;
}
