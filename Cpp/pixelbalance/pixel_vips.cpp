//
// Created by jerry on 2022/5/20.
//
#include "../../include/xmap_util.hpp"

vips::VImage ones(int rows, int cols) {
    return (vips::VImage::black(rows, cols) + 1.0);
}

vips::VImage convolve_vips(const vips::VImage &src, const vips::VImage &kernel, int mode, int blksize) {
//    vips::VImage band, conv_out, def_results;
//    std::vector<vips::VImage> value(3);
//
//    //init
//    def_results = vips::VImage::black(src.width(), src.height());
//    if (mode == 1) {
//        conv_out = src.conv(kernel);
//    } else {
//        for (int i = 0; i < 3; i++) {
//            band = src.extract_band(i);
//            def_results += band.conv(kernel);
//        }
//        auto cut = int((blksize - 1) / 2);
//        conv_out = def_results.extract_area(cut, cut, def_results.width() - cut * 2,
//                                            def_results.height() - cut * 2);
//    }

    vips::VImage conv_out, def_results;
    auto src1 = src.cast(VIPS_FORMAT_DOUBLE);
    if (mode == 1) {
        conv_out = src1.convf(kernel);
    } else {
        def_results = src1.convf(kernel);
        auto cut = int((blksize - 1) / 2);
        conv_out = def_results.extract_area(cut, cut, def_results.width() - cut * 2,
                                            def_results.height() - cut * 2);
    }

    return conv_out;
}

VipsImage *pixel_balance(VipsImage *input, int blk_size) {
    printf("entering pixel balance...\n");
    auto start = x_get_current_ms();

    VipsImage *output;
    vips::VImage in, in1, in2;
    size_t data_size;
    void *buff;
    vips::VImage deta_x, deta_y, deta_x_square, deta_y_square, residual, def_filter,
            nbr_filter, definition_value, mean_nbr, std_nbr, def_value_stat, ref_block, w_s,
            w_m, alpha, beta, dst;


    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in2 = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
    in = in2.cast(VIPS_FORMAT_DOUBLE) / 255;

//    in = vips::VImage(input, vips::NOSTEAL) / 255;
//    auto h = in.height();
//    auto w = in.width();

    deta_x = (in.extract_area(1, 0, w - 1, h)) -
             (in.extract_area(0, 0, w - 1, h));         // [h, w-1, 3]
    deta_y = (in.extract_area(0, 1, w, h - 1)) -
             (in.extract_area(0, 0, w, h - 1));         // [h-1, w, 3]

    deta_x_square = deta_x.extract_area(0, 1, deta_x.width(), deta_x.height() - 1).pow(2.0);
    deta_y_square = deta_y.extract_area(1, 0, deta_y.width() - 1, deta_y.height()).pow(2.0);

    residual = ((deta_x_square + deta_y_square) / 2).pow(0.5);

    def_filter = ones((blk_size - 1), (blk_size - 1)) / pow((blk_size - 1), 2.0);
    nbr_filter = ones(blk_size, blk_size) / pow(blk_size, 2.0);
    definition_value = convolve_vips(residual, def_filter, 0, blk_size);
    mean_nbr = convolve_vips(in, nbr_filter, 1, blk_size);
    std_nbr = (convolve_vips((in - mean_nbr).pow(2.0), nbr_filter, 1, blk_size)).pow(0.5);

    // 找到definition value 的最大像素坐标
    def_value_stat = definition_value.stats();

    auto x_cord = def_value_stat(8, 0)[0];
    auto y_cord = def_value_stat(9, 0)[0];

    ref_block = in.extract_area((int) x_cord, (int) y_cord, blk_size, blk_size);
    vips::VImage b;
    // 计算weights
    std::vector<double> mean_ref, std_ref;
    for (int i = 0; i < 3; i++) {
        b = ref_block.extract_band(i);
        mean_ref.push_back(b.avg());
        std_ref.push_back(b.deviate());
    }

    w_s = std_ref / (std_ref + std_nbr);
    w_m = mean_ref / (mean_ref + mean_nbr);

    alpha = w_s * std_ref / (w_s * std_nbr + (1.0 - w_s) * std_ref);
    beta = w_m * mean_ref + (1.0 - w_m - alpha) * mean_nbr;

    //*********************************//
//    auto cv_alpha = x_vips_to_cv_64f(alpha);
//    auto cv_beta = x_vips_to_cv_64f(beta);
//    auto cv_in = x_vips_to_cv_64f(in);
//    auto arma_alpha = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) cv_alpha);
//    auto arma_beta = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) cv_beta);
//    auto arma_in = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) cv_in);
//    arma::cube arma_dst = arma_alpha % arma_in + arma_beta;
//    auto cv_dst = x_arma_to_cv(arma_dst);
//    dst = x_cv_to_vips_double(cv_dst);
    //*********************************//

    dst = alpha * in + beta;
    auto end = x_get_current_ms();
    printf("pixel balance computation time: %.8f second.\n", (double) (end - start) / 1000);
    printf("done. pixel balance...\n");


    auto cv_dst = x_vips_to_cv_64f(dst);
    dst = x_cv_to_vips_double(cv_dst);
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

   output = dst.get_image();
    // vips_addalpha(dst.get_image(), &output, NULL);

    auto end1 = x_get_current_ms();
    printf("format computation time: %.8f second.\n", (double) (end1 - end) / 1000);
    printf("done. pixel balance...\n");
    return output;

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