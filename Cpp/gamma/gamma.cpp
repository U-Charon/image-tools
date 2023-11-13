//
// Created by jerry on 2022/6/16.
//

#include "../../include/xmap_util.hpp"

VipsImage *image_gamma(VipsImage *input, double alpha, double beta) {
    VipsImage *output;

    auto in = vips::VImage(input, vips::NOSTEAL);
    auto input_zero = (in == 0).ifthenelse(0, 1);
    auto channels = in.bands();
    vips::VImage result[channels], dst;

    in = in / 255.;
    for (int i = 0; i < channels; i++) {
        auto band = in.extract_band(i);
        auto temp = (band > 0).ifthenelse(band, alpha);
        auto x = (beta * (alpha - temp)).exp();
        x = (x == 1).ifthenelse(0, x);
        result[i] = 1 / (1 + x);
        result[i] = (result[i] == 1).ifthenelse(0, result[i]);
    }
    if (channels == 3) {
        auto t2 = x_get_current_ms();
        dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
//        vips_addalpha(dst.get_image(), &output, NULL);
    } else {
        dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
//        output = dst.get_image();
//        g_object_ref(output);
    }


//    x_display_vips_image((in * 255).cast(VIPS_FORMAT_UCHAR), "in", 0);
//    vips_addalpha(in.get_image(), &output, NULL);
    output = dst.get_image();
    g_object_ref(output);
    return output;
}

void AINDANE(const cv::Mat &src, cv::Mat &dst, int sigma1, int sigma2, int sigma3) {
    cv::Mat I;
    cv::cvtColor(src, I, cv::COLOR_BGR2GRAY);

    int histsize = 256;
    float range[] = {0, 256};
    const float *histRanges = {range};
    int bins = 256;
    cv::Mat hist;
    calcHist(&I, 1, nullptr, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);

    int L;
    float cdf = 0;
    int total_pixel = src.rows * src.cols;
    for (int i = 0; i < 256; i++) {
        cdf += hist.at<float>(i) / total_pixel;
        if (cdf >= 0.1) {
            L = i;
            break;
        }
    }

    double z;
    if (L <= 50)
        z = 0;
    else if (L > 150)
        z = 1;
    else
        z = (L - 50) / 100.0;

    cv::Mat I_conv1, I_conv2, I_conv3;
    cv::GaussianBlur(I, I_conv1, cv::Size(0, 0), sigma1, sigma1, cv::BORDER_REFLECT);
    cv::GaussianBlur(I, I_conv2, cv::Size(0, 0), sigma2, sigma2, cv::BORDER_REFLECT);
    cv::GaussianBlur(I, I_conv3, cv::Size(0, 0), sigma3, sigma3, cv::BORDER_REFLECT);


    cv::Mat mean, stddev;
    cv::meanStdDev(I, mean, stddev);
    double global_sigma = stddev.at<double>(0, 0);

    double P;
    if (global_sigma <= 3.0)
        P = 3.0;
    else if (global_sigma >= 10.0)
        P = 1.0;
    else
        P = (27.0 - 2.0 * global_sigma) / 7.0;

    // Look-up table.
    uchar Table[256][256];
    for (int Y = 0; Y < 256; Y++) // Y represents I_conv(x,y)
    {
        for (int X = 0; X < 256; X++) // X represents I(x,y)
        {
            double i = X / 255.0; // Eq.2
            i = (std::pow(i, 0.75 * z + 0.25) + (1 - i) * 0.4 * (1 - z) + std::pow(i, 2 - z)) * 0.5; // Eq.3
            Table[Y][X] = cv::saturate_cast<uchar>(255 * std::pow(i, std::pow((Y + 1.0) / (X + 1.0), P)) + 0.5); // Eq.7 & Eq.8
        }
    }

    dst = src.clone();
    for (int r = 0; r < src.rows; r++) {
        auto *I_it = I.ptr<uchar>(r);
        auto *I_conv1_it = I_conv1.ptr<uchar>(r);
        auto *I_conv2_it = I_conv2.ptr<uchar>(r);
        auto *I_conv3_it = I_conv3.ptr<uchar>(r);
        const auto *src_it = src.ptr<cv::Vec3b>(r);
        auto *dst_it = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; c++) {
            uchar i = I_it[c];
            uchar i_conv1 = I_conv1_it[c];
            uchar i_conv2 = I_conv2_it[c];
            uchar i_conv3 = I_conv3_it[c];
            uchar S1 = Table[i_conv1][i];
            uchar S2 = Table[i_conv2][i];
            uchar S3 = Table[i_conv3][i];
            double S = (S1 + S2 + S3) / 3.0; // Eq.13
//            double S = S1;
            /***
                The following commented codes are original operation(Eq.14) in paper.
            However, the results may contain obvious color spots due to the difference
            between adjacent enhanced luminance is too large.
            Here is an example:
                original luminance     --->     enhanced luminance
                        1              --->             25
                        2              --->             50
                        3              --->             75
            ***/
            //dst_it[c][0] = cv::saturate_cast<uchar>(src_it[c][0] * S / i);
            //dst_it[c][1] = cv::saturate_cast<uchar>(src_it[c][1] * S / i);
            //dst_it[c][2] = cv::saturate_cast<uchar>(src_it[c][2] * S / i);

            /***
                A simple way to deal with above problem is to limit the amplification,
            says, the amplification should not exceed 4 times. You can adjust it by
            yourself, or adaptively set this value.
                You can uncomment(coment) the above(below) codes to see the difference
            and check it out.
            ***/
            double cof = std::min(S / i, 4.0);
            dst_it[c][0] = cv::saturate_cast<uchar>(src_it[c][0] * cof);
            dst_it[c][1] = cv::saturate_cast<uchar>(src_it[c][1] * cof);
            dst_it[c][2] = cv::saturate_cast<uchar>(src_it[c][2] * cof);
        }
    }
}

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
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test1.png)";
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test.tif)";
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test_3Band.TIF)";
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\PAN.tif)";

    const char *f = R"(D:\3.tif)";


    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);

    auto input = vips_image_new_from_file(f, NULL);
    auto src = x_vips_to_cv_8u(input);
    cv::Mat a;
    AGCWD(src, a, 0.5);
    cv::Mat b;
    AINDANE(src, b, 5, 10, 20);

    auto out = image_gamma(input, 0.4, 5.0);
    auto temp = vips::VImage(out, vips::NOSTEAL);
    x_display_vips_image(temp, "out", 0);
    x_display_cv_image(a, "AGCWD", 0);
    x_display_cv_image(b, "AINDANE", 0);
    cv::waitKey();
    cv::destroyAllWindows();
//    save(temp, R"(D:\test_stretch_images\output\test.tif)");
//    save(temp, R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test_out_1band.tif)");
//    save(temp, R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test_out_3bands_1.tif)");
//    save(temp, R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test_out_4bands.tif)");

    g_object_unref(input);
    g_object_unref(out);
}