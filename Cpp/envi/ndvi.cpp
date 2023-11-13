//
// Created by jerry on 2022/10/20.
//
#include "../../include/xmap_util.hpp"

/**
 * 为了让植被之外的地物颜色更加真实，可以只对植被进行增强，这里使用NDVI对植被进行区分。首先先计算NDVI，使用以下波段运算表达式进行加强运算：
    （B3 gt 0.2）*(b2*0.8+b4*0.2)+（B3 le 0.2）*b2  (B3：NDVI)
 *  NDVI=(NIR-R)/(NIR+R)
 *  Green=a*Bandgreen+（1-a）*Bandnir
 */
vips::VImage enhance_green_band(const vips::VImage &in, float ndvi_ratio, float green_ratio) {
    vips::VImage output;
    auto input = in / 255.;

    auto input_zero = (input == 0).ifthenelse(0, 1);

    auto ndvi = (input[3] - input[0]) / (input[3] + input[0]);
    // *********************** 方法1 ************************************
//    auto green_new = (input[1] > ndvi).ifthenelse(input[1], ratio * input[1] + (1 - ratio) * input[3]);

    // *********************** 方法2 ************************************
//    auto nir_greater_then_ratio = (ndvi > ndvi_ratio).ifthenelse(ndvi, 0);
//    auto nir_less_then_ratio = (ndvi < ndvi_ratio).ifthenelse(ndvi, 0);
//    auto green = input[1];
//    auto blue = input[2];
//    auto nir = input[3];
//    auto is_not_green = green * ((ndvi > ndvi_ratio).ifthenelse(0, 1));
//    auto is_green = green * ((ndvi > ndvi_ratio).ifthenelse(1, 0));
////
//////    x_display_vips_image((is_not_green * 255).cast(VIPS_FORMAT_UCHAR), "not", 0);
//////    x_display_vips_image((is_green * 255).cast(VIPS_FORMAT_UCHAR), "yes", 1);
//////
//////    cv::waitKey();
//////    cv::destroyAllWindows();
////
//    auto green_new = nir_greater_then_ratio * (is_green * green_ratio + nir * (1 - green_ratio)) + nir_less_then_ratio * is_green + is_green;
//    output = input[0].bandjoin(green_new + is_not_green).bandjoin(blue).bandjoin(input[3]);

//    green_new = (green_new * (exp(1) -1) + 1).log();
//    auto min = green_new.min();
//    auto max = green_new.max();
//    green_new = 1.0 / 255 + ((green_new - min) * (1.0 - 1.0 / 255)) / (max - min);

//    auto min = blue.min();
//    auto max = blue.max();
//    blue = ((blue - min) / (max - min) * (150 - 1) + 1) / 255;

    // *********************** 方法3 ************************************
    auto greater = (ndvi > ndvi_ratio).ifthenelse(1, 0);
    auto less = (ndvi <= ndvi_ratio).ifthenelse(1, 0);
    auto green_new = greater * (input[1] * green_ratio + input[3] * (1 - green_ratio)) + less * input[1];

    output = input[0].bandjoin(green_new).bandjoin(input[2]).bandjoin(input[3]);


//    auto min = output.min();
//    auto max = output.max();
//    output = 1.0 / 255 + ((output - min) * (1.0 - 1.0 / 255)) / (max - min);

    x_display_vips_image((output * 255).cast(VIPS_FORMAT_UCHAR), "out", 0);

    cv::waitKey();
    cv::destroyAllWindows();

//    ndvi = (ndvi > 0.2).ifthenelse(ndvi, 0);
//    auto green_value = ratio * input[1] + (1 - ratio) * input[3];
//    auto red = input[0];
//    auto green_origin = (input[1] > ndvi).ifthenelse(input[1], 0);
//    auto green = (input[1] > ndvi).ifthenelse(0, green_value*1.3);
//    auto blue = input[2];
//    output = red.bandjoin(green + green_origin).bandjoin(blue);

    // 提取植被
//    ndvi = (ndvi > 0.2).ifthenelse(ndvi, 0);
//    auto green_value = ratio * input[1] + (1 - ratio) * input[3];
//    auto red = (input[1] > ndvi).ifthenelse(0, input[0]);
//    auto green = (input[1] > ndvi).ifthenelse(0, 1);
//    auto blue = (input[1] > ndvi).ifthenelse(0, input[2]);
//    output = red.bandjoin(green).bandjoin(blue);
//    output = output * input_zero;
    return (output * 255).cast(VIPS_FORMAT_UCHAR);
}

vips::VImage enhance_green_band1(const vips::VImage &in, float ratio) {
    vips::VImage output;
    auto input = in / 255.;
    input = (input > 0).ifthenelse(input, 0);

    auto ndvi = (input[3] - input[0]) / (input[3] + input[0]);
    auto nir = input[3];
    auto green = input[1];
    auto condition1 = (ndvi > 0 & ndvi < 0.1).ifthenelse(ndvi, 0);
    auto condition2 = (ndvi > 0.1 & ndvi < 0.2).ifthenelse(ndvi, 0);
    auto condition3 = (ndvi > 0.2 & ndvi < 0.3).ifthenelse(ndvi, 0);
    auto condition4 = (ndvi > 0.3 & ndvi < 0.4).ifthenelse(ndvi, 0);
    auto condition5 = (ndvi > 0.4).ifthenelse(ndvi, 0);
    auto condition6 = (ndvi < 0).ifthenelse(ndvi, 0);

    auto green_new = condition1 * (0.05 * nir + 0.95 * green) +
                     condition2 * (0.1 * nir + 0.9 * green) +
                     condition3 * (0.15 * nir + 0.85 * green) +
                     condition4 * (0.2 * nir + 0.8 * green) +
                     condition5 * (0.25 * nir + 0.75 * green) +
                     condition6 * green + green;

//    auto ndvi = (input[3] - input[0]) / (input[3] + input[0]);
//    auto B3_greater_then_ratio = (ndvi > ratio).ifthenelse(ndvi, 1);
//    auto B3_less_then_ratio = (ndvi < ratio).ifthenelse(ndvi, 1);
//    auto green = input[1];
//    auto nir = input[3];
//    auto green_new = B3_greater_then_ratio * (green * (1-ratio) + nir * ratio) + B3_less_then_ratio * green;

    output = input[0].bandjoin(green_new).bandjoin(input[2]).bandjoin(input[3]);

    return (output * 255).cast(VIPS_FORMAT_UCHAR);
}

vips::VImage enhance_green_band2(const vips::VImage &in, float ratio) {
    vips::VImage output;
    auto input = in / 255.;
    auto input_zero = (input == 0).ifthenelse(0, 1);
    input = (input > 0).ifthenelse(input, 0);
    auto ndvi = (input[3] - input[0]) / (input[3] + input[0]);
    auto green_new = (input[1] > ndvi).ifthenelse(input[1], ratio * input[1] + (1 - ratio) * input[3]);

    // 提取植被
//    auto red = (input[1] > ndvi).ifthenelse(0, input[1]);
//    auto green = (input[1] > ndvi).ifthenelse(0, ratio * input[1] + (1 - ratio) * input[3]);
//    auto blue = (input[1] > ndvi).ifthenelse(0, input[2]);
//    output = red.bandjoin(green).bandjoin(blue);

    output = input[0].bandjoin(green_new).bandjoin(input[2]).bandjoin(input[3]);
    output = output * input_zero;
    return (output * 255).cast(VIPS_FORMAT_UCHAR);
}

vips::VImage ndvi(const vips::VImage &in) {
    auto input = in / 255.;
    auto n = (input[3] - input[0]) / (input[3] + input[0]);

    return n;
}

vips::VImage enhance_green_band3(const vips::VImage &in, float ratio) {
    vips::VImage output;
    auto input = in / 255.;

    auto input_zero = (input == 0).ifthenelse(0, 1);

    auto ndvi = (input[3] - input[0]) / (input[3] + input[0]);
//    auto green = (input[1] > ndvi).ifthenelse(input[1], ratio * input[1] + (1 - ratio) * input[3]);


//    ndvi = (ndvi > 0.2).ifthenelse(ndvi, 0);
    auto green_value = ratio * input[1] + (1 - ratio) * input[3];
    auto red = input[0];
    auto green_origin = (input[1] > ndvi).ifthenelse(input[1], 0);
    auto green_new = (input[1] > ndvi).ifthenelse(0, green_value);
    auto blue = input[2];

//    auto alpha = 0.38, beta=8.0;
//    auto x = (green_new > 0).ifthenelse(green_new, alpha);
//    x = (beta * (alpha - x)).exp();
//    x = (x==1).ifthenelse(0, x);
//    green_new = 1 / (1+x);
//    green_new = (green_new == 1).ifthenelse(0, green_new);

    output = red.bandjoin(green_new + green_origin).bandjoin(blue);

    // 提取植被
//    auto green_value = ratio * input[1] + (1 - ratio) * input[3];
//    auto red = (input[1] > ndvi).ifthenelse(0, input[0]);
//    auto green = (input[1] > ndvi).ifthenelse(0, green_value * 3);
//    auto blue = (input[1] > ndvi).ifthenelse(0, input[2]);
//    output = red.bandjoin(green).bandjoin(blue);
//    output = output * input_zero;
    return (output * 255).cast(VIPS_FORMAT_UCHAR);
}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_cache_set_max(10);
    vips_leak_set(TRUE);

//    const char *img_name = R"(D:\xmap_test_imagedata\green\green.tif)";
//    const char *img_name = R"(D:\xmap_test_imagedata\big\PANSHARPEN_14MAR12025941-P2AS-054063690180_01_P004_UTM_N51-8bit.tif)";
//    const char *img_name = R"(D:\xmap_test_imagedata\output\14MAR12025941-M2AS-054063690180_01_P004.tif)";
//    const char *img_name = R"(D:\xmap_test_imagedata\green\A2.tif)";
    const char *img_name = R"(D:\xmap_test_imagedata\output\minmax.tif)";
    auto img = vips::VImage::new_from_file(img_name);
//    auto n = ndvi(img);
//    n.write_to_file(R"(D:\xmap_test_imagedata\green\ndvi_N51.tif)");
//    n.write_to_file(R"(D:\xmap_test_imagedata\green\ndvi_green.tif)");

//    x_display_vips_image(img, "origin", 0);
//
    auto enhanced = enhance_green_band(img, 0.0, 0.5);
//    enhanced.write_to_file(R"(D:\xmap_test_imagedata\output\minmax加绿.tif)");
//    x_display_vips_image(enhanced, "enhanced", 0);
//    auto enhanced1 = enhance_green_band3(img, 0.2);
//    x_display_vips_image(enhanced1, "enhanced1", 0);

//    auto enhanced1 = enhance_green_band1(img, 0.2);
//    enhanced1.write_to_file(R"(D:\xmap_test_imagedata\output\P004_green.tif)");
//    x_display_vips_image(enhanced1, "enhanced1", 0);
//
//    auto enhanced2 = enhance_green_band2(img, 0.2);
//    x_display_vips_image(enhanced1, "enhanced2", 0);
//
//    cv::waitKey();
//    cv::destroyAllWindows();

    return 0;
}