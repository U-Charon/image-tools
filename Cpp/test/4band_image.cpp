//
// Created by jerry on 2022/10/18.
//

#include "../../include/xmap_util.hpp"
#include "../api/xmap.h"

int find_low(const vips::VImage &in, double T) {
    auto hist = in.hist_find();
    printf("T: %f\n", T);
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

void optimized_stretch_batch_no_calculate1(char *input_name, char *output_name, double min_percent, double max_percent,
                                           double min_adjust, double max_adjust, int order) {
    printf("entering optimize stretch batch NO calculate process...\n");
    auto start = x_get_current_ms();

    vips::VImage input, band, dst;
    input = vips::VImage::new_from_file(input_name);

    auto channels = input.bands();
    vips::VImage result[channels];

    printf("%s, [c=%d, order=%d]\n", input_name, channels, order);

    if (order == 1) {
        if (channels == 3) {
            input = input[2].bandjoin(input[1]).bandjoin(input[0]);
        } else if (channels == 4) {
            input = input[2].bandjoin(input[1]).bandjoin(input[0]).bandjoin(input[3]);
        }
    }

//    double a, b, c, d, min_value = 1 / 255., max_value = 254 / 255.;
    double c, d, min_value = 1 / 255., max_value = 1.0;
    auto input_zero = (input == 0).ifthenelse(0, 1);

    auto total_non_zero_pixel = x_count_non_zero_and_white(input);
    printf("none zero=%f\n", total_non_zero_pixel);


    if (channels == 1) {
        auto a = find_low(input, min_percent * total_non_zero_pixel);
        auto b = find_high(input, (1 - max_percent) * total_non_zero_pixel);
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
            auto a = find_low(band, min_percent * total_non_zero_pixel);
            auto b = find_high(band, (1 - max_percent) * total_non_zero_pixel);
            printf("none zero=%f, a=%d, b=%d\n", total_non_zero_pixel, a, b);
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
    printf("[min=%f, max=%f, avg=%f]\n", dst.min(), dst.max(), dst.avg());
//    dst.tiffsave(output_name, vips::VImage::option()->
//            set("compression", VIPS_FOREIGN_TIFF_COMPRESSION_NONE)->
//            set("Q", 75));
    dst.write_to_file(output_name, vips::VImage::option()->
            set("compression", VIPS_FOREIGN_TIFF_COMPRESSION_NONE)->
            set("Q", 100));

//    g_object_unref(img);

    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
    printf("done. optimize batch stretch NO calculate...\n");
}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_cache_set_max(10);
    vips_leak_set(TRUE);

//    char *img_file = R"(D:\xmap_test_imagedata\4bands\test.tif)";
//    char *out1 = R"(D:\xmap_test_imagedata\output\test.tif)";
    char *img_file = R"(D:\xmap_test_imagedata\20221012\N50.tif)";
    char *out1 = R"(D:\xmap_test_imagedata\output\N50.tif)";


    auto img = vips::VImage::new_from_file(img_file);

    printf("[%d %d %d]\n", img.width(), img.height(), img.bands());

    optimized_stretch_batch_no_calculate1(img_file, out1, 0.025, 0.99, 0.3, 0.3, 1);

//    unsigned long int total = 2214380246;
//    long double a = (long double) total * 0.3;
//    printf("total=%lu, a=%Lf\n", total, a);

//    auto img1 = vips_image_new_from_file(img_file, NULL);
//    VipsImage *out_img;
//    optimized_stretch(img1, &out_img, 0.025, 0.99, 0.3, 0.3, 1);
//    double avg, min, max;
//    vips_avg(out_img, &avg, NULL);
//    vips_min(out_img, &min, NULL);
//    vips_max(out_img, &max, NULL);
//    printf("[min=%f, max=%f, avg=%f]\n", min, max, avg);
//    vips_image_write_to_file(out_img, R"(D:\xmap_test_imagedata\output\test2.tif)", NULL);
//
//    g_object_unref(img1);
//    g_object_unref(out_img);

    return 0;
}