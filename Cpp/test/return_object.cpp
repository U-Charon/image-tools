//
// Created by Zeng Yu on 2022/7/24.
//
#include "../../include/xmap_util.hpp"

VipsImage *fn1(VipsImage *in, int a) {
    VipsImage *output;

    auto input = vips::VImage(in, vips::NOSTEAL);
    auto dst = input * a / 255;

    vips_addalpha(dst.get_image(), &output, NULL);

    return output;
}

void fn2(VipsImage *in, VipsImage **output, int a) {
    auto input = vips::VImage(in, vips::NOSTEAL);
    auto dst = input * a / 255;

    vips_addalpha(dst.get_image(), *(&output), NULL);

}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_cache_set_max(10);
    vips_leak_set(TRUE);

    const char *img_file = "D:\\xmap_test_imagedata\\raw1.tif";
    auto img = vips_image_new_from_file(img_file, NULL);

    // Test case 1: fn1, see weather there has memory leak
    auto out = fn1(img, 200);
    g_object_unref(out);

    // Test case 2: fn2, see weather there has memory leak
    VipsImage *out1;
    fn2(img, &out1, 200);
    g_object_unref(out1);


    g_object_unref(img);

    return 0;
}