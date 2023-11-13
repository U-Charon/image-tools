//
// Created by jerry on 2022/5/10.
//
#include "../../include/xmap_util.hpp"


//cv::Mat get_dark_channel_img(const cv::Mat &src, const int radius) {
//    int height = src.rows;
//    int width = src.cols;
//    cv::Mat darkChannelImg(src.size(), CV_8UC1);
//    cv::Mat darkTemp(darkChannelImg.size(), darkChannelImg.type());
//
//    //求取src中每个像素点三个通道中的最小值，将其赋值给暗通道图像中对应的像素点
//    for (int i = 0; i < height; i++) {
//        const auto *srcPtr = src.ptr<uchar>(i);
//        auto *dstPtr = darkTemp.ptr<uchar>(i);
//        for (int j = 0; j < width; j++) {
//            int b = srcPtr[3 * j];
//            int g = srcPtr[3 * j + 1];
//            int r = srcPtr[3 * j + 2];
//            dstPtr[j] = cv::min(cv::min(b, g), r);
//        }
//    }
//
//    //把图像分成patch,求patch框内的最小值,得到dark_channel image
//    //r is the patch radius, patchSize=2*r+1
//    //这一步实际上是最小值滤波的过程
//    cv::Mat rectImg;
//    int patchSize = 2 * radius + 1;
//    for (int j = 0; j < height; j++) {
//        for (int i = 0; i < width; i++) {
//            cv::getRectSubPix(darkTemp, cv::Size(patchSize, patchSize), cv::Point(i, j), rectImg);
//            double minValue = 0;
//            cv::minMaxLoc(rectImg, &minValue, nullptr, nullptr, nullptr); //get min pix value
//            darkChannelImg.at<uchar>(j, i) = cv::saturate_cast<uchar>(minValue);//using saturate_cast to set pixel value to [0,255]
//        }
//    }
//
////    displayCVmat(darkChannelImg, "dark", 0);
//    return darkChannelImg;
//}

double get_global_light(const cv::Mat &darkChannelImg) {
    //这里是简化的处理方式,A的最大值限定为220
    double minAtomsLight = 220;//经验值
    double maxValue = 0;
//    cv::Point maxLoc;
    minMaxLoc(darkChannelImg, nullptr, &maxValue, nullptr, nullptr);
    double A = cv::min(minAtomsLight, maxValue);

    return A;
}

cv::Mat get_trans_img(const cv::Mat &darkChannelImg, const double A, double w) {
    cv::Mat transmissionImg(darkChannelImg.size(), CV_8UC1);
    cv::Mat look_up(1, 256, CV_8UC1);

    uchar *look_up_ptr = look_up.data;
    for (int k = 0; k < 256; k++) {
        look_up_ptr[k] = cv::saturate_cast<uchar>(255 * (1 - w * k / A));
    }

    cv::LUT(darkChannelImg, look_up, transmissionImg);

    return transmissionImg;
}

cv::Mat fastGuidedFilter(const cv::Mat &I_org, const cv::Mat &p_org, int r, double eps, int s) {
    /*
    % GUIDED FILTER   O(N) time implementation of guided filter.
    %
    %   - guidance image: I (should be a gray-scale/single channel image)
    %   - filtering input image: p (should be a gray-scale/single channel image)
    %   - local window radius: r
    %   - regularization parameter: eps
    */

    cv::Mat I, _I;
    I_org.convertTo(_I, CV_64FC1, 1.0 / 255);
    resize(_I, I, cv::Size(), 1.0 / s, 1.0 / s, 1);

    cv::Mat p, _p;
    p_org.convertTo(_p, CV_64FC1, 1.0 / 255);
    //p = _p;
    resize(_p, p, cv::Size(), 1.0 / s, 1.0 / s, 1);

    //[hei, wid] = size(I);
    int hei = I.rows;
    int wid = I.cols;

    r = (2 * r + 1) / s + 1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4

    //mean_I = boxfilter(I, r) ./ N;
    cv::Mat mean_I;
    cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));

    //mean_p = boxfilter(p, r) ./ N;
    cv::Mat mean_p;
    cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));

    //mean_Ip = boxfilter(I.*p, r) ./ N;
    cv::Mat mean_Ip;
    cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));

    //cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    //mean_II = boxfilter(I.*I, r) ./ N;
    cv::Mat mean_II;
    cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

    //var_I = mean_II - mean_I .* mean_I;
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);

    //a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
    cv::Mat a = cov_Ip / (var_I + eps);

    //b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
    cv::Mat b = mean_p - a.mul(mean_I);

    //mean_a = boxfilter(a, r) ./ N;
    cv::Mat mean_a;
    cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
    cv::Mat rmean_a;
    resize(mean_a, rmean_a, cv::Size(I_org.cols, I_org.rows), 1);

    //mean_b = boxfilter(b, r) ./ N;
    cv::Mat mean_b;
    cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
    cv::Mat rmean_b;
    resize(mean_b, rmean_b, cv::Size(I_org.cols, I_org.rows), 1);

    //q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
    cv::Mat q = rmean_a.mul(_I) + rmean_b;
    cv::Mat q1;
    q.convertTo(q1, CV_8UC1, 255, 0);

    return q1;
}

cv::Mat get_dehazed_channel(cv::Mat srcChannel, cv::Mat transmissionChannel, double A) {
    double tmin = 0.1;
    double tmax;

    cv::Mat dehazedChannel(srcChannel.size(), CV_8UC1);
    for (int i = 0; i < srcChannel.rows; i++) {
        for (int j = 0; j < srcChannel.cols; j++) {
            double transmission = transmissionChannel.at<uchar>(i, j);

            tmax = (transmission / 255) < tmin ? tmin : (transmission / 255);
            //(I-A)/t +A
            dehazedChannel.at<uchar>(i, j) = cv::saturate_cast<uchar>(abs((srcChannel.at<uchar>(i, j) - A) / tmax + A));
        }
    }
    return dehazedChannel;
}

cv::Mat get_dehazed_img_guided_filter(const cv::Mat &src, const cv::Mat &darkChannelImg, double ratio, int r, double eps) {
    cv::Mat dehazedImg = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);

    cv::Mat transmissionImg(src.rows, src.cols, CV_8UC3);
    cv::Mat fineTransmissionImg(src.rows, src.cols, CV_8UC3);
    std::vector<cv::Mat> srcChannel, dehazedChannel, transmissionChannel, fineTransmissionChannel;

    cv::split(src, srcChannel);
    double A = get_global_light(darkChannelImg);

    cv::split(transmissionImg, transmissionChannel);
    auto trans_image = get_trans_img(darkChannelImg, A, ratio);
    transmissionChannel[0] = trans_image;
    transmissionChannel[1] = trans_image;
    transmissionChannel[2] = trans_image;

    cv::split(fineTransmissionImg, fineTransmissionChannel);
    fineTransmissionChannel[0] = fastGuidedFilter(srcChannel[0], transmissionChannel[0], r, eps, 8);
    fineTransmissionChannel[1] = fastGuidedFilter(srcChannel[1], transmissionChannel[1], r, eps, 8);
    fineTransmissionChannel[2] = fastGuidedFilter(srcChannel[2], transmissionChannel[2], r, eps, 8);

//    cv::merge(fineTransmissionChannel, fineTransmissionImg);
//    cv::imshow("fineTransmissionChannel", fineTransmissionImg);

    cv::split(dehazedImg, dehazedChannel);
    dehazedChannel[0] = get_dehazed_channel(srcChannel[0], fineTransmissionChannel[0], A);
    dehazedChannel[1] = get_dehazed_channel(srcChannel[1], fineTransmissionChannel[1], A);
    dehazedChannel[2] = get_dehazed_channel(srcChannel[2], fineTransmissionChannel[2], A);

    cv::merge(dehazedChannel, dehazedImg);

    return dehazedImg;
}

cv::Mat dark_channel_image(const cv::Mat &src, const int &radius) {
    int height = src.rows; // 1880
    int width = src.cols; // 1936

    int patchSize = 2 * radius + 1;
    arma::Mat<uchar> darkChannelImg(width + int(patchSize / 2), height + int(patchSize / 2)), final, sub_mat;
    cv::Mat padded_cv;

    cv::copyMakeBorder(src, padded_cv, (int) patchSize / 2, (int) patchSize / 2, (int) patchSize / 2,
                       (int) patchSize / 2, cv::BORDER_REFLECT);
    auto armaSrc = x_cv_to_arma_cube((cv::Mat_<cv::Vec<uchar, 3>>) padded_cv);
    darkChannelImg = arma::min(armaSrc, 2);


    arma::Cube<uchar> result(width, height, patchSize * patchSize);
    int i = 0;

    for (int m = 0; m < patchSize; m++) {
        for (int n = 0; n < patchSize; n++) {
//            printf("m=%d, n=%d, [%d, %d]\n", m, n, darkChannelImg.n_cols, darkChannelImg.n_rows);
            sub_mat = darkChannelImg.submat(m, n, m + width - 1, n + height - 1);
//            printf("sub_mat [%d, %d]\n", sub_mat.n_cols, sub_mat.n_rows);
//            printf("slice_mat [%d, %d]\n", result.slice(i).n_cols, result.slice(i).n_rows);
            result.slice(i) = sub_mat;
            ++i;
        }
    }

    final = arma::min(result, 2);
    auto out = x_arma_to_cv_single(final);
//    displayCVmat(out, "mat", 0);

    return out.clone();
}

int main(int argc, char **argv) {
//    const char *f = R"(D:\test_stretch_images\raw1.tif)";
    const char *f = R"(D:\test_stretch_images\Tile_C1.tif)";

    int radius = 7, diameter = 10;
    double ratio = 0.95, eps = 0.01;

    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);
    auto input = vips::VImage::new_from_file(f);
    if (input.bands() == 4) {
        input = input[0].bandjoin(input[1]).bandjoin(input[2]);
    }

    auto start = x_get_current_ms();
    // 1. === cv
//    auto src0 = vips_to_cv_8u(input);
//    auto darkChanelImg0 = get_dark_channel_img(src0, radius);
//    auto out0 = get_dehazed_img_guided_filter(src0, darkChanelImg0, ratio);
//    auto dst0 = matToVipsDouble(out0);
    // 2. === arma
    auto src = x_vips_to_cv_8u(input);
    auto darkChanelImg = dark_channel_image(src, radius);
    auto out = get_dehazed_img_guided_filter(src, darkChanelImg, ratio, 256, eps);
    auto out1 = get_dehazed_img_guided_filter(src, darkChanelImg, ratio, 8, eps);
    auto dst = x_cv_to_vips_double(out);
    auto dst1 = x_cv_to_vips_double(out1);
    auto end = x_get_current_ms();
    printf("computation time %.8f second.\n", (double) (end - start) / 1000);

    x_display_vips_image(dst.cast(VIPS_FORMAT_UCHAR), "dst", 0);
    x_display_vips_image(dst1.cast(VIPS_FORMAT_UCHAR), "dst1", 1);
//    x_display_cv_image(darkChanelImg, "dark", 0);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}