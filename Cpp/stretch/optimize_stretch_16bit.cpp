//
// Created by jerry on 2022/5/24.
//
#include "../../include/xmap_util.hpp"
#include "../../Cpp/api/xmap.h"

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

    double max_total = 0;
    int j = w - 1;
    while (max_total <= T) {
        max_total += hist(j, 0)[0];
        j -= 1;
    }
    auto high = j + 1;

    return high;
}

void save(vips::VImage &input, const char *filename) {
    input.write_to_file(filename, vips::VImage::option()->
            set("compression", VIPS_FOREIGN_TIFF_COMPRESSION_NONE)->
            set("Q", 100));
}

int main(int argc, char **argv) {
    const char *f = "/Users/Jerry/dev/20220401-balancetest/test.tif";
//    const char *f = R"(C:\Users\zengy\GolandProjects\xmapSR\test_64bit_small.tif)";
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test.tif)";
//    const char *f = R"(C:\Users\zengy\GolandProjects\xmapSR\hanshou_zy3.tif)";

    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);
    auto start = x_get_current_ms();

    auto input = vips::VImage::new_from_file(f);
    input = input[2].bandjoin(input[1]).bandjoin(input[0]);

//    if (input.interpretation() != VIPS_INTERPRETATION_RGB16) {
//        printf("your image is not 16bit.");
//        return -1;
//    }

    // ---------------------------------- optimized stretch -------------------------------------------//
    double a, b, c, d;
    vips::VImage band, dst, mid, stretch, sigmoid1, sigmoid2, result[3], os[3], sig1[3];
    /*
     * beta1 越大，稍微暗的部分变暗， 稍微亮的部分变亮；
     * alpha 控制变化的区间
     */
    double min_percent = 0.01, max_percent = 0.99, min_adjust = 0.1, max_adjust = 0.3,
            min_value = 1.0 / 255, max_value = 254.0 / 255, alpha = 0.4, beta1 = 6, beta2 = 2;

    for (int i = 0; i < 3; i++) {
        band = input.extract_band(i);
//        auto mean_band = band.avg();

        a = find_low(band, min_percent * band.height() * band.width());
        b = find_high(band, (1 - max_percent) * band.height() * band.width());
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

//        printf("a=%f, b=%f, c=%f, d=%f, min=%f, max=%f\n", a, b, c, d);
        band = (band < c).ifthenelse(min_value, band);
        auto v1 = (band - c) / (d - c) * (max_value - min_value) + min_value;
        band = (band >= c & band <= d).ifthenelse(v1, band);

        result[i] = (band > d).ifthenelse(max_value, band);
        os[i] = result[i];
        // ---------------------------------- gamma adjust ---------------------------------------------//
//        auto gamma = -0.3 / log10(result[i].avg());
//        printf("gamma = %f\n", gamma);
//        result[i] = result[i].pow(gamma);
//        sig[i] = result[i];
        // ---------------------------------- sigmoid contrast 1---------------------------------------------//
        auto meanv = result[i].avg();
        auto x = (beta1 * (alpha - result[i])).exp();
        sig1[i] = 1 / (1 + x);
        // ---------------------------------- sigmoid contrast 1---------------------------------------------//

    }

    stretch = os[0].bandjoin(os[1]).bandjoin(os[2]);
    stretch = (stretch * 255).cast(VIPS_FORMAT_UCHAR);

    // ---------------------------------- sigmoid contrast 2---------------------------------------------//
    auto u = (beta2 / 255.) * stretch - (beta2 / 2); // 对于input_data[0, 255] ----> u[0, beta]-beta/2 ----> [-beta/2, beta/2]
    std::vector<vips::VImage> r;
    for (int i = 0; i < stretch.bands(); i++) {
        auto meanv = u[i].avg();  // meanv: mid
        auto aa = (u[i] - meanv).exp();
        auto bb = (meanv - u[i]).exp();
        auto sigma = (aa - bb) / (aa + bb); // sigma ∈ [-1, 1]  本质雨stretch_1, streth_2有区别；
        r.push_back((sigma + 1) / 2);
    }
    // ---------------------------------- sigmoid contrast 2---------------------------------------------//

    dst = r[0].bandjoin(r[1]).bandjoin(r[2]);
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

//    auto end = x_get_current_ms();
//    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);


    sigmoid1 = sig1[0].bandjoin(sig1[1]).bandjoin(sig1[2]);
    sigmoid1 = (sigmoid1 * 255).cast(VIPS_FORMAT_UCHAR);



//    save(dst, R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test_final.tif)");
//    save(dst, R"(C:\Users\zengy\CLionProjects\xmapBalance\img\chengdu_stretch.tif)");
//    auto end1 = x_get_current_ms();
//    printf("total save tif time %.8f second.\n", (double) (end1 - end) / 1000);
//    save(stretch, R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test_optimize.tif)");
//    x_display_vips_image((input).cast(VIPS_FORMAT_UCHAR), "o", 0);

    x_display_vips_image(dst, "sigmoid2", 2);
    x_display_vips_image(stretch, "os", 0);
    x_display_vips_image(sigmoid1, "sigmoid1", 1);
//    x_display_vips_image(sigmoid2, "sigmoid2", 1);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}