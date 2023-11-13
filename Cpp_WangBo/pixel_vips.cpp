//
// Created by jerry on 2022/5/20.
//
#include "../../include/xmap_util.hpp"

vips::VImage ones(int rows, int cols) {
    return (vips::VImage::black(rows, cols) + 1.0);
}

vips::VImage convolve_vips(const vips::VImage &src, const vips::VImage &kernel) {
    vips::VImage conv_out = vips::VImage::black(src.width(), src.height());
    int kernel_param = floor((double) (kernel.width() - 1.0) / 2 * 0.1);
    auto src1 = src.cast(VIPS_FORMAT_DOUBLE);

    for (int b = 0; b < 3; b++) {
        conv_out += src1[b].conv(kernel, vips::VImage::option()->
                set("layers", kernel_param)->
                set("cluster", kernel_param));
    }

    return conv_out;
}

VipsImage *pixel_balance(VipsImage *input, int blk_size) {
    printf("entering pixel balance...\n");
    auto start = x_get_current_ms();

    VipsImage *output;
    vips::VImage in, in1, in2;
    int cut = (blk_size - 1) / 2;
    int radius = floor((double) (blk_size - 1.0) / 2 * 0.1);
    size_t data_size;
    void *buff;


    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in2 = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
    in = in2.cast(VIPS_FORMAT_DOUBLE) / 255;

    in = vips::VImage(input, vips::NOSTEAL) / 255;
    in = in.cast(VIPS_FORMAT_DOUBLE);
    auto h = in.height();
    auto w = in.width();

    auto deta_x = (in.extract_area(1, 0, w - 1, h)) -
                  (in.extract_area(0, 0, w - 1, h));         // [h, w-1, 3]
    auto deta_y = (in.extract_area(0, 1, w, h - 1)) -
                  (in.extract_area(0, 0, w, h - 1));         // [h-1, w, 3]

    auto deta_x_square = deta_x.extract_area(0, 1, deta_x.width(), deta_x.height() - 1).pow(2.0);
    auto deta_y_square = deta_y.extract_area(1, 0, deta_y.width() - 1, deta_y.height()).pow(2.0);

    auto residual = ((deta_x_square + deta_y_square) / 2).pow(0.5);

    auto def_filter = ones((blk_size - 1), (blk_size - 1)) / pow((blk_size - 1), 2.0);
    auto nbr_filter = ones(blk_size, blk_size) / pow(blk_size, 2.0);
//    auto definition_value = convolve_vips(residual, def_filter, blk_size);
//    mean_nbr = convolve_vips(in, nbr_filter, 1, blk_size);
//    std_nbr = (convolve_vips((in - mean_nbr).pow(2.0), nbr_filter, 1, blk_size)).pow(0.5);
    auto def_v_tmp = convolve_vips(residual, def_filter);
    auto mean_nbr = in.conv(nbr_filter,
                            vips::VImage::option()->
                                    set("layers", radius)->
                                    set("cluster", radius));
    auto std_nbr = (in - mean_nbr).pow(2.0).conv(nbr_filter,
                                                 vips::VImage::option()->
                                                         set("layers", radius)->
                                                         set("cluster", radius)).pow(0.5);
    auto definition_value = def_v_tmp.extract_area(cut, cut,
                                                   residual.width() - cut,
                                                   residual.height() - cut);

    // 找到definition value 的最大像素坐标
    auto def_value_stat = definition_value.stats();

    auto x_cord = def_value_stat(8, 0)[0];
    auto y_cord = def_value_stat(9, 0)[0];
    auto start_x = x_cord - cut;
    auto start_y = y_cord - cut;
    vips::VImage ref_block;

    if (start_x > 0 && start_y > 0) {
        ref_block = in.extract_area((int) start_x, (int) start_y, blk_size, blk_size);
    } else {
        ref_block = in.extract_area((int) x_cord, (int) y_cord, blk_size, blk_size);
    }

    // 计算weights
    std::vector<double> mean_ref, std_ref;
    for (int i = 0; i < 3; i++) {
        mean_ref.push_back(ref_block[i].avg());
        std_ref.push_back(ref_block[i].deviate());
    }

    auto w_s = std_ref / (std_ref + std_nbr);
    auto w_m = mean_ref / (mean_ref + mean_nbr);
    auto alpha = w_s * std_ref / (w_s * std_nbr + (1.0 - w_s) * std_ref);
    auto beta = w_m * mean_ref + (1.0 - w_m - alpha) * mean_nbr;

    vips::VImage result = vips::VImage::black(w, h);
    result = alpha * in + beta;
    result = (result * 255).cast(VIPS_FORMAT_UCHAR);
    result = result.bandjoin(255);
    auto dst = result;

//    auto end = x_get_current_ms();
//    printf("pixel balance computation time: %.8f second.\n", (double) (end - start) / 1000);
//    printf("done. pixel balance...\n");

//    buff = result.write_to_memory(&data_size);
//    auto dst = vips::VImage::new_from_memory_steal(buff, data_size, w, h,
//                                                   result.bands(), VIPS_FORMAT_UCHAR);

//    auto cv_dst = x_vips_to_cv_64f(dst);
//    dst = x_cv_to_vips_double(cv_dst);

    auto temp = dst.get_image();
    g_object_ref(temp);
    vips_addalpha(temp, &output, NULL);
    g_object_unref(temp);
    g_object_ref(output);

   output = dst.get_image();
    // vips_addalpha(dst.get_image(), &output, NULL);

    auto end1 = x_get_current_ms();
    printf("total computation time: %.8f second.\n", (double) (end1 - start) / 1000);
    printf("done. pixel balance...\n");

    return output;
}

int main(int argc, char **argv) {
    const char *f = "/Users/Jerry/dev/CLionProjects/xmapImageBalance/img/test1.png";
//    const char *f = "/Users/Jerry/dev/CLionProjects/xmapImageBalance/img/5m.tif";
    int blk = 31;

    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);
//    auto input = vips::VImage::new_from_file(f);
    auto input = vips_image_new_from_file(f, NULL);
    auto vin = vips::VImage(input, vips::NOSTEAL);

    // ********* 测试1次调用
//    auto out = pixel_balance(input, blk);
//    auto vou = vips::VImage(out, vips::NOSTEAL);
//    vou = vou[0].bandjoin(vou[1]).bandjoin(vou[2]);
    // ********* 测试1次调用
    // ++++++++++++++++++++++++++++++++++++
    // ********* 循环调用5次，每次都用上次返回值
    auto out = pixel_balance(input, blk);
    auto vou = vips::VImage(out, vips::NOSTEAL);
    for (int i = 0; i < 5; i++) {
        out = pixel_balance(vou.get_image(), blk);
        vou = vips::VImage(out, vips::NOSTEAL);
        vou = vou[0].bandjoin(vou[1]).bandjoin(vou[2]);
    }
    // ********* 循环调用5次，每次都用上次返回值

//    x_display_vips_image(vin, "o", 0);
    x_display_vips_image(vou, "r", 0);
    cv::waitKey();
    cv::destroyAllWindows();

    g_object_unref(input);
    g_object_unref(out);

    vips_shutdown();

}
int main(int argc, char **argv) {

//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test1.png)";
    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\5m.tif)";
    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);
    auto start = x_get_current_ms();
    auto in = vips_image_new_from_file(f);

    auto out = pixel_balance(in, 51);
    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);

    g_object_unref(in);
    vips_shutdown();
    return 0;
}