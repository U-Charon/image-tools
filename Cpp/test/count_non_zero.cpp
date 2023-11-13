//
// Created by Zeng Yu on 2022/7/22.
//

#include "../../include/xmap_util.hpp"

int count_non_zero(const vips::VImage &in) {
    auto img_hist = in.hist_find();
    auto total_count = in.width() * in.height();
    auto zero_count = int(img_hist(0, 0)[0]);

    return total_count - zero_count;
}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_cache_set_max(10);
    vips_leak_set(TRUE);

    const char *img_file = "/Users/Jerry/dev/20220401-balancetest/test.TIF";
//    const char *img_file = "/Users/Jerry/dev/20220401-balancetest/raw1.tif";
//    const char *img_file = "/Users/Jerry/dev/20220401-balancetest/test2-2.tif";
//    const char *img_file = "/Users/Jerry/dev/20220401-balancetest/test5.png";

    auto img = vips::VImage::new_from_file(img_file);
    img = img[2].bandjoin(img[1]).bandjoin(img[0]);
    auto img_hist = img.hist_find();
    printf("hist w,h [%d, %d]\n", img_hist.width(), img_hist.height());

    auto zero_count = int(img_hist(0, 0)[0]);
    auto total_count = img.width() * img.height();
    printf("VIPS1: zero=%d, total=%d, non-zeo=%d\n", zero_count, total_count, total_count - zero_count);

    int vips_zero = 0;
    for (int i = 0; i < 3; i++) {
        auto hist = img[i].hist_find();
        vips_zero += int(hist(0, 0)[0]);
    }
    vips_zero = vips_zero / 3;
    printf("VIPS2: zero=%d, total=%d, non-zeo=%d\n", vips_zero, total_count, total_count - vips_zero);

    int cv_non_zero = 0;
    for (int i = 0; i < 3; i++) {
        auto cv_img = x_vips_to_cv_single(img[i]);
        cv_non_zero += cv::countNonZero(cv_img);
    }
    cv_non_zero = cv_non_zero / 3;
    printf("CV1: zero=%d, total=%d, non-zeo=%d\n", total_count - cv_non_zero, total_count, cv_non_zero);

    auto img_greater_zero = (img > 0).ifthenelse(1, img);
    auto one = int(img_greater_zero.hist_find()(1, 0)[0]);
    printf("VIPS: channel non-zero [%d]\n", one);

    return 0;
}
