//
// Created by jerry on 2022/10/25.
//
#include "../../include/xmap_util.hpp"

void AGCWD(const cv::Mat &src, cv::Mat &dst, double alpha) {
    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();
    int total_pixels = rows * cols;

    cv::Mat L;
    cv::Mat HSV;
    std::vector<cv::Mat> HSV_channels;
    if (channels == 1) {
        L = src.clone();
    } else {
        cv::cvtColor(src, HSV, cv::COLOR_BGR2HSV_FULL);
        cv::split(HSV, HSV_channels);
        L = HSV_channels[2];
    }

    int histsize = 256;
    float range[] = {0, 256};
    const float *histRanges = {range};
    int bins = 256;
    cv::Mat hist;
    calcHist(&L, 1, nullptr, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);

    double total_pixels_inv = 1.0 / total_pixels;
    cv::Mat PDF = cv::Mat::zeros(256, 1, CV_64F);
    for (int i = 0; i < 256; i++) {
        PDF.at<double>(i) = hist.at<float>(i) * total_pixels_inv;
    }

    double pdf_min, pdf_max;
    cv::minMaxLoc(PDF, &pdf_min, &pdf_max);
    cv::Mat PDF_w = PDF.clone();
    for (int i = 0; i < 256; i++) {
        PDF_w.at<double>(i) = pdf_max * std::pow((PDF_w.at<double>(i) - pdf_min) / (pdf_max - pdf_min), alpha);
    }

    cv::Mat CDF_w = PDF_w.clone();
    double culsum = 0;
    for (int i = 0; i < 256; i++) {
        culsum += PDF_w.at<double>(i);
        CDF_w.at<double>(i) = culsum;
    }
    CDF_w /= culsum;

    std::vector<uchar> table(256, 0);
    for (int i = 1; i < 256; i++) {
        table[i] = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, 1 - CDF_w.at<double>(i)));
    }

    cv::LUT(L, table, L);

    if (channels == 1) {
        dst = L.clone();
    } else {
        cv::merge(HSV_channels, dst);
        cv::cvtColor(dst, dst, cv::COLOR_HSV2BGR_FULL);
    }

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
    const char *img_name = R"(D:\xmap_test_imagedata\ndwi\minmax.tif)";
    auto img = vips::VImage::new_from_file(img_name);

    auto src = x_vips_to_cv_8u(img);
    cv::Mat out;
    AGCWD(src, out, 0.2);
    auto dst = x_cv_to_vips_double(out);
    dst = dst.cast(VIPS_FORMAT_UCHAR);
    dst.write_to_file(R"(D:\xmap_test_imagedata\output\agcwd0.tif)");
    x_display_vips_image(dst, "agcwd", 0);

//    img = img / 255.;
//    auto ndwi = (img[1] - img[3]) / (img[1] + img[3]);
////    auto R = img[0] * (ndwi > 0.2).ifthenelse(ndwi, 0);
////    auto G = img[1] * (ndwi > 0.2).ifthenelse(ndwi, 0);
////    auto B = img[2] * (ndwi > 0.2).ifthenelse(ndwi, 0);
//    auto water = img * (ndwi > 0.2).ifthenelse(1, 0);
//    dst = dst.bandjoin(img[3]);
//    auto temp1 = (dst/255) * (ndwi < 0.3).ifthenelse(1, 0);
//    auto temp2 = (dst/255) * (ndwi < 0.3).ifthenelse(0, 1);
//    auto output = (water + temp2)/2 + temp1;
//    output = (output*255).cast(VIPS_FORMAT_UCHAR);
//    x_display_vips_image((temp * 255).cast(VIPS_FORMAT_UCHAR), "ndwi", 0);
//    x_display_vips_image((water * 255).cast(VIPS_FORMAT_UCHAR), "water", 0);
//    x_display_vips_image(output, "dst", 0);

    dst.write_to_file(R"(D:\xmap_test_imagedata\output\agcwd1.tif)");
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}