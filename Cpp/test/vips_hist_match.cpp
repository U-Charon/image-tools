//
// Created by jerry on 2022/10/27.
//
#include "../../include/xmap_util.hpp"

vips::VImage zero_hist() {
//    cv::Mat hist(1, 256, CV_64FC1);
//    for (int i = 0; i < 256; i++) {
//        if (i == 0) {
//            hist.at<double>(0, i) = 0;
//        } else {
//            hist.at<double>(0, i) = 1;
//        }
//    }
//    return x_cv_to_vips_double(hist);
    double a[256];

    for (int i = 0; i < 256; i++) {
        i == 0 ? a[i] = 0.0 : a[i] = 1.0;
    }
    auto out = vips::VImage::new_matrix(256, 1, a, 256);

    return out;
}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_cache_set_max(10);
    vips_leak_set(TRUE);

//    const char *ref_name = R"(D:\xmap_test_imagedata\mapping\yingshe\dt\dt.tif)";
    const char *img_name = R"(D:\xmap_test_imagedata\mapping\yingshe\ys\A2.tif)";

//    const char *ref_name = R"(D:\xmap_test_imagedata\mapping\ref.jpg)";
//    const char *img_name = R"(D:\xmap_test_imagedata\mapping\img.jpg)";

    const char *ref_name = R"(D:\xmap_test_imagedata\mapping\yingshe\ref1.tif)";
//    const char *img_name = R"(D:\xmap_test_imagedata\mapping\yingshe\raw.tif)";

    auto img = vips::VImage::new_from_file(img_name);
    auto ref = vips::VImage::new_from_file(ref_name);
    vips::VImage out_bands[3];
    vips::VImage band4;

    if (img.bands() == 4) {
        band4 = img[3];
    }

//    for (int i = 0; i < 3; i++) {
//        auto band_hist = img[i].hist_find();
//
////        auto temp = zero_hist();
////        printf("band hist 0 value count before=%f\n", band_hist(0, 0)[0]);
////        band_hist = band_hist * temp;
////        *VIPS_MATRIX(band_hist.get_image(), 0, 0) = 0;
////        printf("band hist 0 value count after=%f\n", band_hist(0, 0)[0]);
//
//        auto img_hist_norm = band_hist.hist_cum().hist_norm();
//        auto ref_hist_norm = ref[i].hist_find().hist_cum().hist_norm();
////        printf("img hist norm bands=%d, w=%d, h=%d\n", img_hist_norm.bands(), img_hist_norm.width(), img_hist_norm.height());
////        printf("img hist norm min=%f, max=%f, avg=%f\n", img_hist_norm.min(), img_hist_norm.max(), img_hist_norm.avg());
//        auto lut = img_hist_norm.hist_match(ref_hist_norm);
//        img_hist_norm
//        out_bands[i] = (img[i]).maplut(lut);
//    }
//    auto out = out_bands[0].bandjoin(out_bands[1]).bandjoin(out_bands[2]);
//    img = img[0].bandjoin(img[1]).bandjoin(img[2]);

//    auto img_hist_norm = img.hist_find().hist_cum().hist_norm();
//    auto ref_hist_norm = ref.hist_find().hist_cum().hist_norm();
//    auto lut = img_hist_norm.hist_match(ref_hist_norm);
//    printf("lut bands=%d,%d,%d\n", lut.bands(), lut.width(), lut.height());
//    for (int i = 0; i < 256; i++) {
//        printf("lut  value=[%f, %f, %f]\n", lut[0](i,0)[0], lut[1](i,0)[0], lut[2](i,0)[0]);
//
//        if (lut(i, 0)[0] == 1.0) {
//            *VIPS_MATRIX(lut.get_image(), i, 0) = img_hist_norm(i, 0)[0];
////            printf("after %d, lut=%f, img_hist=%f\n", i, lut(i, 0)[0], img_hist_norm(i, 0)[0]);
//        }
//    }


//    out = out.cast(VIPS_FORMAT_UCHAR);
//    x_display_vips_image(out, "out", 0);
//    x_display_vips_image(lut.cast(VIPS_FORMAT_UCHAR), "lut", 0);
//    out.cast(VIPS_FORMAT_UCHAR).write_to_file(R"(D:\xmap_test_imagedata\mapping\yingshe\output1.tif)");
//    cv::waitKey();
//    cv::destroyAllWindows();
    vips::VImage result[3], img_cdf, ref_cdf;
    double l[256];
    int lookup_value = 0;

    int flag = 0;
    for (int b = 0; b < 3; b++) {
        auto img_hist = img[b].hist_find();
        auto ref_hist = ref[b].hist_find();
        auto temp = zero_hist();
        img_hist = img_hist * temp;
        ref_hist = ref_hist * temp;

        auto img_hist_cum = img_hist.hist_cum();
        auto ref_hist_cum = ref_hist.hist_cum();
        img_cdf = (img_hist_cum / img_hist_cum.max());
        ref_cdf = (ref_hist_cum / ref_hist_cum.max());

        auto start = x_get_current_ms();
        auto c = x_vips_to_arma_mat(img_cdf);
        auto d = x_vips_to_arma_mat(ref_cdf);
        for (int i = 0; i < 256; i++) {
            auto src_value = c[i];
            for (int j = 0; j < 256; j++) {
                auto ref_value = d[j];
                if (ref_value >= src_value) {
                    lookup_value = j;
                    break;
                }
            }
            l[i] = lookup_value;
        }
        auto end = x_get_current_ms();
        printf("time = %f\n", double(end - start) / 1000);
        auto lut = vips::VImage::new_matrix(256, 1, l, 256);
        result[b] = img[b].maplut(lut);
    }
//    auto start = x_get_current_ms();
////    auto a = x_vips_to_arma_mat(img_cdf);
////    auto b = x_vips_to_arma_mat(ref_cdf);
//
//    for (int i = 0; i < 256; i++) {
//
////        auto src_value = img_cdf(i, 0)[0];
//        auto src_value = a[i];
//
//        for (int j = 0; j < 256; j++) {
////            auto ref_value = ref_cdf(j, 0)[0];
//            auto ref_value = b[j];
//
//            if (ref_value >= src_value) {
//                lookup_value = j;
//                break;
//            }
//        }
//        l[i] = lookup_value;
//    }
//    auto end = x_get_current_ms();
//    printf("time = %f\n", double(end - start) / 1000);
//    auto lut = vips::VImage::new_matrix(256, 1, l, 256);
//    result[0] = img[0].maplut(lut);

//    for (int i = 0; i < 256; i++) {
////            printf("%.9f\t", (double)img_hist_cum(i, 0)[0] / img_hist_cum.max());
////            printf("%.9f\t", (double)img_hist_cum(i, 0)[0]);
////            flag += 1;
////            if (flag % 8 == 0) {
////                printf("\n");
//        auto src_value = img_cdf(i, 0)[0];
//        for (int j = 0; j < 256; j++) {
//            auto ref_value = ref_cdf(j, 0)[0];
//            if (ref_value >= src_value) {
//                lookup_value = j;
//                break;
//            }
//        }
//
//        lut[i] = lookup_value;
//    }
//
//    for (double &i: lut) {
//        printf("%f   ", i);
//        flag += 1;
//        if (flag % 10 == 0) {
//            printf("\n");
//        }
//    }


    auto out = result[0].bandjoin(result[1]).bandjoin(result[2]);
    out = out.cast(VIPS_FORMAT_UCHAR);
    x_display_vips_image(out, "out", 0);
    out.write_to_file(R"(D:\xmap_test_imagedata\mapping\yingshe\A2-v.tif)");
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}