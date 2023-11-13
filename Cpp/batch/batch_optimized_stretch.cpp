//
// Created by jerry on 2022/6/10.
//
#include <filesystem>
#include "../../include/xmap_util.hpp"

void save(vips::VImage &input, const char *filename) {
    input.write_to_file(filename, vips::VImage::option()->
            set("compression", VIPS_FOREIGN_TIFF_COMPRESSION_NONE)->
            set("Q", 100));
}

int find_low(const vips::VImage &in, int band, double T) {
    auto hist = in[band].hist_find();

    double min_total = 0;
    int i;
    if (in.min() == 0) {
        i = 1;
    } else { i = 0; }

    while (min_total <= T) {
        min_total += hist(i, 0)[0];
        i += 1;
    }
    auto low = i - 1;

    return low;
}

int find_high(const vips::VImage &in, int band, double T) {
    auto hist = in[band].hist_find();
    auto w = hist.width();

    double max_total = 0;
//    int j = w - 1;
    int j;
    if (in.max() == 255 || in.max() == 65535) {
        j = w - 2;
    } else { j = w - 1; }
    while (max_total <= T) {
        max_total += hist(j, 0)[0];
        j -= 1;
    }
    auto high = j + 1;

    return high;
}

void calculate_hist_total1(const char *directory, int bands, int order, double min_p, double max_p, int *a, int *b) {
    vips::VImage img;
    int pixel_number = 0;
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
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), tolower);
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
            pixel_number += img.width() * img.height();
        }
    }

    // 找到边界的a、b值
    T1 = min_p * pixel_number;
    T2 = (1 - max_p) * pixel_number;

    for (int m = 0; m < bands; m++) {
        min_total = 0.0;
        max_total = 0.0;

        if (bands == 1) {
            // --- find a:
            if (hist_total[0].min() == 0) {
                i = 1;
            } else { i = 0; }
            while (min_total <= T1) {
                min_total += hist_total[m](i, 0)[0];
                i += 1;
            }
            a[m] = i - 1; // a值
            // --- find b:
            auto w = hist_total[m].width();
            if ((hist_total[0].min() == 255) || (hist_total[0].min() == 65535)) {
                j = w - 2;
            } else { j = w - 1; }
            while (max_total <= T2) {
                max_total += hist_total[m](j, 0)[0];
                j -= 1;
            }
            b[m] = j + 1; // b值
        } else {
            // --- find a:
            if ((hist_total[0].min() == 0) && (hist_total[1].min() == 0) && (hist_total[2].min() == 0)) {
                i = 1;
            } else { i = 0; }
            while (min_total <= T1) {
                min_total += hist_total[m](i, 0)[0];
                i += 1;
            }
            a[m] = i - 1; // a值
            // --- find b:
            auto w = hist_total[m].width();
            if (((hist_total[0].max() == 255) && (hist_total[1].max() == 255) && (hist_total[2].max() == 255)) ||
                ((hist_total[0].max() == 65535) && (hist_total[1].max() == 65535) && (hist_total[2].max() == 65535))) {
                j = w - 2;
            } else { j = w - 1; }
            while (max_total <= T2) {
                max_total += hist_total[m](j, 0)[0];
                j -= 1;
            }
            b[m] = j + 1; // b值
        }
    }
}

vips::VImage optimized_stretch_batch1(const vips::VImage &in, double min_adjust, double max_adjust, int order, int channels, const int *low, const int *high) {
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

    double a, b, c, d, min_value = 1 / 255., max_value = 254 / 255.;
    auto input_zero = (input == 0).ifthenelse(0, 1);

    if (channels == 1) {
        a = low[0];
        b = high[0];
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        input = (input < c).ifthenelse(min_value, input);
        auto v1 = (input - c) / (d - c) * (max_value - min_value) + min_value;
        input = (input >= c & input <= d).ifthenelse(v1, input);

        dst = ((input > d) & (input < 65535)).ifthenelse(max_value, input);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
    } else {
        for (int i = 0; i < channels; i++) {
            band = input.extract_band(i);
            // test
            auto a1 = find_low(input, i, 0.025 * band.height() * band.width());
            auto b1 = find_high(input, i, 0.01 * band.height() * band.width());
            printf("--- band %d, low/high[%d, %d]\n", i, a1, b1);
            // test
            a = low[i];
            b = high[i];
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min_value, band);
            auto v1 = (band - c) / (d - c) * (max_value - min_value) + min_value;
            band = ((band >= c) & (band <= d)).ifthenelse(v1, band);
            //TODO 这里如果有可能要判断是否是8位和16位，来去掉255的值（如果是8位白边的话）
            result[i] = ((band > d) & (band < 65535)).ifthenelse(max_value, band);
        }
    }

    switch (channels) {
        case 3:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        case 4:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        default:
            break;
    }

    return dst;
}

int main(int argc, char **argv) {
//    const char *in_dir = R"(D:\test_stretch_images\stretch16bit)";
    const char *in_dir = R"(D:\test_stretch_images\input)";
    const char *out_dir = R"(D:\test_stretch_images\output\test.tif)";
    int band_number = 4;
    int low[band_number], high[band_number];

    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(true);
    calculate_hist_total1(in_dir, band_number, 1, 0.025, 0.99, low, high);


    for (int i = 0; i < band_number; i++) {
        printf("band %d, low/high[%d, %d]\n", i, low[i], high[i]);
    }

    for (const auto &entry: std::filesystem::directory_iterator(in_dir)) {
        auto filename = entry.path().string();
        size_t pos = filename.find('.', 0);
        auto suffix = filename.substr(pos + 1, 3);
        transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
        if (suffix == "tif") {
            auto img = vips::VImage::new_from_file(filename.c_str());
            auto out = optimized_stretch_batch1(img, 0.3, 0.3, 1, band_number, low, high);
            save(out, out_dir);
        }
    }

    return 0;
}
