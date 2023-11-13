//
// Created by jerry on 2022/10/24.
//
#include "../../include/xmap_util.hpp"

vips::VImage vips_padding(const vips::VImage &in, int top, int bottom, int left, int right) {
    auto top_left = in.extract_area(0, 0, left, top);
    auto top_right = in.extract_area(in.width() - right, 0, right, top);
    auto bottom_left = in.extract_area(0, in.height() - bottom, left, bottom);
    auto bottom_right = in.extract_area(in.width() - right, in.height() - bottom, right, bottom);

    auto top_area = in.extract_area(0, 0, in.width(), top);
    auto bottom_area = in.extract_area(0, in.height() - bottom, in.width(), bottom);
    auto left_area = in.extract_area(0, 0, left, in.height());
    auto right_area = in.extract_area(in.width() - right, 0, right, in.height());

    vips::VImage image_padding = vips::VImage::black(in.width() + left + right, in.height() + top + bottom);

    image_padding = image_padding.insert((top_left.flip(VIPS_DIRECTION_HORIZONTAL)).flip(VIPS_DIRECTION_VERTICAL), 0, 0);
    image_padding = image_padding.insert((top_right.flip(VIPS_DIRECTION_HORIZONTAL)).flip(VIPS_DIRECTION_VERTICAL), image_padding.width() - right, 0);
    image_padding = image_padding.insert(in, left, top);
    image_padding = image_padding.insert(top_area.flip(VIPS_DIRECTION_VERTICAL), left, 0);
    image_padding = image_padding.insert(left_area.flip(VIPS_DIRECTION_HORIZONTAL), 0, top);
    image_padding = image_padding.insert(right_area.flip(VIPS_DIRECTION_HORIZONTAL), image_padding.width() - right, top);
    image_padding = image_padding.insert(bottom_area.flip(VIPS_DIRECTION_VERTICAL), left, image_padding.height() - bottom);
    image_padding = image_padding.insert((bottom_left.flip(VIPS_DIRECTION_HORIZONTAL)).flip(VIPS_DIRECTION_VERTICAL), 0, image_padding.height() - bottom);
    image_padding = image_padding.insert((bottom_right.flip(VIPS_DIRECTION_HORIZONTAL)).flip(VIPS_DIRECTION_VERTICAL), image_padding.width() - right,
                                         image_padding.height() - bottom);


    return image_padding;
}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_cache_set_max(10);
    vips_leak_set(TRUE);

    const char *img_name = R"(D:\xmap_test_imagedata\big\green.tif)";

    auto img = vips::VImage::new_from_file(img_name);
    auto padding = vips_padding(img, 100, 100, 100, 100);

    x_display_vips_image(img, "origin", 0);
    x_display_vips_image(padding, "padding", 0);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}