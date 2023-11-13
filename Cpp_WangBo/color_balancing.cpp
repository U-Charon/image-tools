//
// Created by jerry on 2022/5/19.
//
#include "../../include/xmap_util.hpp"

vips::VImage vips_ones(int rows, int cols) {
    return (vips::VImage::black(rows, cols) + 1.0);
}

arma::mat convolve_vips_mat(const arma::cube &src, const vips::VImage &kernel) {
    auto cv_src = x_arma_to_cv((arma::Cube<double>) src);
    auto vips_src = x_cv_to_vips_double(cv_src);

    vips::VImage total = vips::VImage::black(vips_src.width(), vips_src.height());
    auto vips_conv = vips_src.convf(kernel);
    for (int i = 0; i < 3; i++) {
        total += vips_src[i].convf(kernel);
    }

    auto cv_conv = x_vips_to_cv_single(total);
    auto arma_conv = x_cv_to_arma_mat<double>((cv::Mat_<double>) cv_conv);

    return arma_conv;
}

arma::cube convolve_vips_cube(const arma::cube &src, const vips::VImage &kernel) {
    auto cv_src = x_arma_to_cv(src);
    auto vips_src = x_cv_to_vips_double(cv_src);

    auto vips_conv = vips_src.convf(kernel);

    auto cv_conv = x_vips_to_cv_64f(vips_conv);
    auto arma_conv = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) cv_conv);

    return arma_conv;
}

vips::VImage pixel_balance(const vips::VImage &input, int blk_size) {
    using namespace arma;

    auto in = input.cast(VIPS_FORMAT_DOUBLE);
    auto h = input.height();
    auto w = input.width();
    int cut = (blk_size - 1) / 2;

    auto cv_in = x_vips_to_cv_64f(in);
    auto arma_in = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) cv_in);

    cube deta_x = arma_in.tube(1, 0, w - 1, h - 1) -
                  arma_in.tube(0, 0, w - 2, h - 1); // [row x col] 475x476
    cube deta_y = arma_in.tube(0, 1, w - 1, h - 1) -
                  arma_in.tube(0, 0, w - 1, h - 2); // [row x col] 476x475

    cube deta_x_square = pow(deta_x.tube(0, 1, deta_x.n_rows - 1, deta_x.n_cols - 1), 2.0);
    cube deta_y_square = pow(deta_y.tube(1, 0, deta_y.n_rows - 1, deta_y.n_cols - 1), 2.0);
    cube residual = pow(((deta_x_square + deta_y_square) / 2), 0.5);

    mat def_filter = ones((blk_size - 1), (blk_size - 1)) / pow((blk_size - 1), 2.0);
    mat nbr_filter = ones(blk_size, blk_size) / pow(blk_size, 2.0);

    mat definition_value = zeros((w - 1) - (blk_size - 1), (h - 1) - (blk_size - 1));
    cube mean_nbr = zeros(w, h, 3);
    cube std_nbr = zeros(w, h, 3);

    mat def_temp(residual.n_rows, residual.n_cols);

    auto start = x_get_current_ms();
//    for (int i = 0; i < 3; i++) {
//        def_temp += arma::conv2(residual.slice(i), def_filter, "same");
//        mean_nbr.slice(i) = arma::conv2(arma_in.slice(i), nbr_filter, "same");
//        std_nbr.slice(i) = pow(arma::conv2(pow((arma_in.slice(i) - mean_nbr.slice(i)), 2.0),
//                                           nbr_filter, "same"), 0.5);
//
//    }
//    def_temp.submat(0, 0, 10, 10).print("def_temp_arma:");
//    std_nbr.tube(0, 0, 10, 10).print("mean_nbr_arma:");
//    std_nbr.tube(0, 0, 10, 10).print("std_nbr_arma:");
    //---------------------------------------------------------------------//
    auto def_filter_vips = vips_ones((blk_size - 1), (blk_size - 1)) / pow((blk_size - 1), 2.0);
    auto nbr_filter_vips = vips_ones(blk_size, blk_size) / pow(blk_size, 2.0);
//    def_temp = convolve_vips_cube(residual, def_filter_vips);
    def_temp = convolve_vips_mat(residual, def_filter_vips);
//    def_temp = sum(def_temp_cube, 2);


//    def_temp.submat(0, 0, 10, 10).print("def_temp_vips:");
    mean_nbr = convolve_vips_cube(arma_in, nbr_filter_vips);
    std_nbr = pow(convolve_vips_cube(pow((arma_in - mean_nbr), 2.0), nbr_filter_vips), 0.5);
//    def_temp.submat(0, 0, 10, 10).print("def_temp_vips:");
//    std_nbr.tube(0, 0, 10, 10).print("mean_nbr_vips:");
//    std_nbr.tube(0, 0, 10, 10).print("std_nbr_vips:");
    //---------------------------------------------------------------------//

    auto end = x_get_current_ms();
    printf("conv time %.8f second.\n", (double) (end - start) / 1000);


    definition_value = def_temp.submat(cut, cut, residual.n_rows - cut, residual.n_cols - cut);

    rowvec b = max(definition_value, 0);
    colvec c = max(definition_value, 1);
    uword y = index_max(b) - cut;
    uword x = index_max(c) - cut;
    cube ref_block = arma_in.tube(x - cut, y - cut, size(blk_size, blk_size));

    colvec mean_ref = mean(mean(ref_block, 0), 0);
    cube std_cube = reshape(ref_block, blk_size * blk_size, 3, 1);
    rowvec std_ref = stddev(std_cube.slice(0));

    cube w_s(w, h, 3),
            w_m(w, h, 3), alpha(w, h, 3),
            beta(w, h, 3);

    for (int i = 0; i < 3; i++) {
        w_s.slice(i) = std_ref(i) / (std_ref(i) + std_nbr.slice(i));
        w_m.slice(i) = mean_ref(i) / (mean_ref(i) + mean_nbr.slice(i));

        alpha.slice(i) = w_s.slice(i) * std_ref(i) / (w_s.slice(i) % std_nbr.slice(i) + (1.0 - w_s.slice(i)) * std_ref(i));
        beta.slice(i) = w_m.slice(i) * mean_ref(i) + (1.0 - w_m.slice(i) - alpha.slice(i)) % mean_nbr.slice(i);
    }


    cube arma_dst = alpha % arma_in + beta;


    auto cv_dst = x_arma_to_cv(arma_dst);
    vips::VImage dst = x_cv_to_vips_double(cv_dst);

//    auto cv_dst = x_arma_to_cv(std_nbr);
//    vips::VImage dst = x_cv_to_vips_double(cv_dst);
    return dst;
}

int main(int argc, char **argv) {

//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test1.png)";
    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\5m.tif)";
    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);
    auto start = x_get_current_ms();
    auto in = vips::VImage::new_from_file(f);

    auto out = pixel_balance(in, 51);

    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
//    x_display_vips_image(out.cast(VIPS_FORMAT_UCHAR), "result", 1);
//    x_display_vips_image(in, "in", 0);
//    cv::waitKey();
//    cv::destroyAllWindows();
}