//
// Created by jerry on 2022/5/13.
//
#include "../../include/xmap_util.hpp"
#include "../../Cpp/api/xmap.h"

vips::VImage calculate_local_mean(const vips::VImage &input, double p, double overlap) {
    const double constant = 128.0 / 45;  // 理想状态下的mean/std
    auto h = input.height();
    auto w = input.width();
    auto im_mean = input.avg();
    auto im_std = input.deviate();

    auto rho = p / im_std * im_mean / constant;
    auto adw_size = int(sqrt(rho * h * rho * w) / 2) * 2 + 1;
    auto adw_stride = int(adw_size * (1 - overlap) / 2) * 2;

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
    if (padding_top < int(adw_size / 2)) {
        padding_top += int(adw_stride / 2);
        padding_bottom += int(adw_stride / 2);
        num_h += 1;
    }
    if (padding_left < int(adw_size / 2)) {
        padding_left += int(adw_stride / 2);
        padding_right += int(adw_stride / 2);
        num_w += 1;
    }

    auto img_padding = x_cv_padding(input, padding_top, padding_bottom, padding_left, padding_right);
    int pad_h = img_padding.rows;
    int pad_w = img_padding.cols;
    auto arma_padding = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) img_padding);
    img_padding.release();
//    printf("LMM: arma_padding shape [%llu, %llu, %f, %f]\n", arma_padding.n_rows, arma_padding.n_cols,
//           arma_padding.min(), arma_padding.max());

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

    auto m_h_ = pad_h - (adw_size - 1);
    auto m_w_ = pad_w - (adw_size - 1);

    auto top_x = int((m_w_ - w) / 2);
    auto top_y = int((m_h_ - h) / 2);
    auto idx = vips::VImage::xyz(m_w_, m_h_);
    idx = idx / adw_stride;
    idx = idx.extract_area(top_x, top_y, w, h);

    auto cv_lmm = x_arma_to_cv(local_mean_map);
    auto result = x_cv_to_vips_double(cv_lmm).mapim(idx);
    cv_lmm.release();
//    x_display_cv_image(cv_lmm, "lmm", 0);
//    x_display_vips_image((result*255).cast(VIPS_FORMAT_UCHAR), "tcm", 0);
//    printf("result min, max [%f, %f]\n", result.min(), result.max());
//    cv::waitKey();
//    cv::destroyAllWindows();

    return result;
}

vips::VImage polynomial_fitting(const vips::VImage &input, double p, double overlap) {
    vips::VImage out;

//    printf("input image min=%.8f, max=%.8f, bands=%d, avg=%.8f\n", input.min(), input.max(), input.bands(), input.avg());
//    auto cv_input = x_vips_to_cv_64f(input);
//    printf("cv image w=%d, h=%d, type=%d, ch=%d\n", cv_input.cols, cv_input.rows, cv_input.type(), cv_input.channels());
//    auto arma_input = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) cv_input);
//    auto arma_stats = x_arma_cube_stats(arma_input);
//    printf("arma image min=%f, max=%f, mean=%.8f, [%llu, %llu, %llu]\n", arma_stats[1], arma_stats[2],
//           arma_stats[0], arma_input.n_rows, arma_input.n_cols, arma_input.n_slices);

    const double constant = 128.0 / 45;  // 理想状态下的mean/std
    auto h = input.height();
    auto w = input.width();
    auto im_mean = input.avg();
    auto im_std = input.deviate();

    auto rho = p / im_std * im_mean / constant;
    auto adw_size = int(sqrt(rho * h * rho * w) / 2) * 2 + 1;
    auto adw_stride = int(adw_size * (1 - overlap) / 2) * 2;

    auto num_h = int(((h - adw_size) / adw_stride) + 1) + 1;
    auto num_w = int(((w - adw_size) / adw_stride) + 1) + 1;

    // 剩余部分padding填充之后 再补adw_stride 但不能确保adw的中心点落在原img的外围
    auto padding_h = adw_size + adw_stride * (num_h - 1) - h + adw_stride;
    auto padding_w = adw_size + adw_stride * (num_w - 1) - w + adw_stride;
    num_h += 1;
    num_w += 1;

    auto padding_top = int(padding_h / 2);
    auto padding_bottom = padding_h - padding_top;
    auto padding_left = int(padding_w / 2);
    auto padding_right = padding_w - padding_left;
    if (padding_top < int(adw_size / 2)) {
        padding_top += int(adw_stride / 2);
        padding_bottom += int(adw_stride / 2);
        num_h += 1;
    }
    if (padding_left < int(adw_size / 2)) {
        padding_left += int(adw_stride / 2);
        padding_right += int(adw_stride / 2);
        num_w += 1;
    }

//    printf("mean=%.10f, std=%.10f, rho=%f, adw_size=%d, adw_stride=%d\n", im_mean, im_std, rho, adw_size, adw_stride);
//    printf("padding [%d, %d, %d, %d]\n", padding_top, padding_bottom, padding_left, padding_right);
    auto img_padding = x_cv_padding(input, padding_top, padding_bottom, padding_left, padding_right);
    auto arma_padding = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) img_padding);
    img_padding.release();
//    auto stat1 = x_arma_cube_stats(arma_padding);
//    printf("arma image min=%f, max=%f, mean=%.8f, [%llu, %llu, %llu]\n", stat1[1], stat1[2],
//           stat1[0], arma_padding.n_rows, arma_padding.n_cols, arma_padding.n_slices);
    arma::cube local_mean_map(num_w, num_h, 3), lmm;
    arma::cube local_mean_center(num_w, num_h, 2), lmc;
    arma::cube tube;
//    arma::vec center_wh;
    double center_h, center_w;

    for (int m = 0; m < num_h; m++) {
        auto adw_top = m * adw_stride;
        auto adw_bottom = adw_top + adw_size;
        center_h = adw_top + int(adw_size / 2) + 1;
        for (int n = 0; n < num_w; n++) {
            auto adw_left = n * adw_stride;
            auto adw_right = adw_left + adw_size;
            center_w = adw_left + int(adw_size / 2) + 1;
            tube = arma_padding.tube(adw_left, adw_top, adw_right - 1, adw_bottom - 1);
            local_mean_map.tube(n, m) = arma::mean(arma::mean(tube, 0), 1);
//            center_wh = {center_w, center_h};
            local_mean_center.tube(n, m) = (arma::vec) {center_w, center_h};
//            local_mean_center.tube(n, m)(0) = center_w;
//            local_mean_center.tube(n, m)(1) = center_h;
        }
    }

//    auto a = x_arma_to_cv(local_mean_map);
//    x_display_cv_image(a, "lmm", 0);
//    cv::waitKey();

    lmm = local_mean_map.tube(1, 1, local_mean_map.n_rows - 1, local_mean_map.n_cols - 1);
    lmc = local_mean_center.tube(1, 1, local_mean_center.n_rows - 1, local_mean_center.n_cols - 1);
    lmc.slice(0) -= padding_top;
    lmc.slice(1) -= padding_left;
//    lmc.print("LMC:");

    arma::vec t_y = arma::linspace(0, h, h + 1);
    arma::mat t_y_mat = arma::reshape(t_y, 1, h);
    arma::mat t_idx_0 = arma::repmat(t_y_mat, w, 1);

    arma::vec t_x = arma::linspace(0, w, w + 1);
    arma::mat t_x_mat = arma::reshape(t_x, w, 1);
    arma::mat t_idx_1 = arma::repmat(t_x_mat, 1, h);
//    printf("index shape [%llu, %llu]\n", t_idx_0.n_rows, t_idx_0.n_cols);

    arma::vec x1 = arma::vectorise(lmc.slice(0));
    arma::vec x2 = arma::vectorise(lmc.slice(1));
//    printf("x1: [133.0, 1933.0] ==== %f, %f\n", x1.min(), x1.max());
//    printf("x2: [78.0, 2928.0] ==== %f, %f\n", x2.min(), x2.max());

    arma::vec x0 = arma::ones<arma::vec>(size(x1));
    arma::mat x = arma::join_rows(x0, x1);
    x = arma::join_rows(x, x2);

    arma::vec x_1 = arma::vectorise(t_idx_0);
    arma::vec x_2 = arma::vectorise(t_idx_1);
    arma::vec x_0 = arma::ones<arma::vec>(size(x_1));
    arma::mat xx = arma::join_rows(x_0, x_1);
    xx = arma::join_rows(xx, x_2);

    arma::vec x3 = arma::pow(x1, 2);
    arma::vec x4 = arma::pow(x2, 2);
//    printf("x4: [6084.0, 8573184.0] ==== %f, %f\n", x4.min(), x4.max());
    arma::vec x5 = x1 % x2;
    x = arma::join_rows(x, x3);
    x = arma::join_rows(x, x4);
    x = arma::join_rows(x, x5);

    arma::vec x_3 = arma::pow(x_1, 2);
    arma::vec x_4 = arma::pow(x_2, 2);

    arma::vec x_5 = x_1 % x_2;
    xx = arma::join_rows(xx, x_3);
    xx = arma::join_rows(xx, x_4);
    xx = arma::join_rows(xx, x_5);

    arma::vec x6 = arma::pow(x1, 3);
    arma::vec x7 = arma::pow(x2, 3);
//    printf("x7: [474552.0, 25102282752.0] ==== %f, %f\n", x7.min(), x7.max());
    arma::vec x8 = x1 % x4;
    arma::vec x9 = x2 % x3;
//    printf("x9: [1379742.0, 10940439792.0] ==== %f, %f\n", x9.min(), x9.max());
    x = arma::join_rows(x, x6);
    x = arma::join_rows(x, x7);
    x = arma::join_rows(x, x8);
    x = arma::join_rows(x, x9);

    arma::vec x_6 = arma::pow(x_1, 3);
    arma::vec x_7 = arma::pow(x_2, 3);
    arma::vec x_8 = x_1 % x_4;
    arma::vec x_9 = x_2 % x_3;

//    printf("x_9: 12780674907 ==== %f, %f\n", x_9.min(), x_9.max());
    xx = arma::join_rows(xx, x_6);
    xx = arma::join_rows(xx, x_7);
    xx = arma::join_rows(xx, x_8);
    xx = arma::join_rows(xx, x_9);
//    printf("xx: [0 , 27081081027] ==== %f, %f\n", xx.min(), xx.max());

    arma::cube y = arma::reshape(lmm, lmm.n_rows * lmm.n_cols, 3, 1);
    arma::mat alpha = arma::inv(x.t() * x) * x.t() * y.slice(0);
//    printf("alpha: [-2.9912186530210407e-05 ,0.27389336525549196] ==== %f, %f\n", alpha.min(), alpha.max());

    arma::mat target_surface = xx * alpha;
//    printf("target_surface shape [%llu, %llu]\n", target_surface.n_rows, target_surface.n_cols);
//    target_surface.print("target_surface:");

    arma::mat r = arma::reshape(target_surface.col(0), w, h);
    arma::mat g = arma::reshape(target_surface.col(1), w, h);
    arma::mat b = arma::reshape(target_surface.col(2), w, h);
    arma::cube dst = arma::join_slices(r, g);
    dst = arma::join_slices(dst, b);
    auto stat = x_arma_cube_stats(dst);
//    printf("[%f, %f, %f]\n", stat[0], stat[1], stat[2]);

//    dst.print("DST:");

    cv::Mat cv_result = x_arma_to_cv(dst);
    out = x_cv_to_vips_double(cv_result);
//    cv::Mat t;
//    cv_result.convertTo(t, CV_8UC3);
//    x_display_cv_image(t, "surface", 0);
//    cv::waitKey();
//    cv::destroyAllWindows();
    cv_result.release();

    return out;
}

vips::VImage dodging_third_order(const vips::VImage &input, const vips::VImage &tar,
                                 double p, double overlap, double alpha) {
    vips::VImage target;

    if (tar.bands() != 3) {
        target = tar[0].bandjoin(tar[1]).bandjoin(tar[2]);
    } else {
        target = tar;
    }
    auto scale_opt = vips::VImage::option()->set("vscale", (double) input.height() / target.height());
    scale_opt->set("kernel", VIPS_KERNEL_LANCZOS3);
    target = target.resize((double) input.width() / target.width(), scale_opt);
//    printf("target wh=[%d, %d]\n", target.width(), target.height());

    auto local_mean_map = calculate_local_mean(input, p, overlap);
    auto target_color_map = polynomial_fitting(target, p, overlap);

//    x_display_vips_image((local_mean_map*255).cast(VIPS_FORMAT_UCHAR), "lmm", 0);
//    x_display_vips_image((target_color_map*255).cast(VIPS_FORMAT_UCHAR), "tcm", 0);
//    cv::waitKey();
//    cv::destroyAllWindows();

    auto gamma = target_color_map.log() / local_mean_map.log();
    auto dst = alpha * input.pow(gamma);
//    printf("[...dst shape, min, max, avg], [%d, %d, %f, %f, %f]\n", dst.width(),
//           dst.height(), dst.min(), dst.max(), dst.avg());
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

//    x_display_vips_image((dst * 255.).cast(VIPS_FORMAT_UCHAR), "r", 0);
//    cv::waitKey();
//    cv::destroyAllWindows();

    return dst;
}

int main(int argc, char **argv) {
    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\5m.tif)";
    const char *t = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\5m_target1.tif)";

    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);
    auto input = vips::VImage::new_from_file(f);
    auto target = vips::VImage::new_from_file(t);

    auto start = x_get_current_ms();
//    auto out = dodging_third_order(input / 255, target / 255, 0.1, 0.2, 1.0);
    auto out = dodging_third_order(input / 255, target / 255, 0.1, 0.2, 1.0);
    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);

    x_display_vips_image(out, "r", 0);
    cv::waitKey();
    cv::destroyAllWindows();

    // 测试函数循环5次
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test1.png)";
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\5m.tif)";
//    if (vips_init(argv[0])) {
//        vips_error_exit("unable to init vips");
//    }
//    vips_leak_set(TRUE);
////    auto input = vips::VImage::new_from_file(f);
//    size_t size0, size1;
//    auto input = vips_image_new_from_file(f, NULL);
////    vips_image_write_to_memory(input, &size0);
////    printf("1---- %d\n", size0);
//    VipsImage *out, *out1, *out2, *out3, *out4;
//    out = pixel_balance(input, 31);
////    vips_image_write_to_memory(out, &size1);
////    printf("2---- %d\n", size1);
//    out1 = pixel_balance(out, 31);
//
//    out2 = pixel_balance(out1, 31);
//    out3 = pixel_balance(out2, 31);
////    printf(vips_image_get_history(out));
//    out4 = pixel_balance(out3, 31);
//    g_object_unref(input);
////    g_object_unref(out1);
////    g_object_unref(out2);
////    g_object_unref(out3);
//    vips_shutdown();


    return 0;
}