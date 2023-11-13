//
// Created by jerry on 2022/9/5.
//
#include "../../include/xmap_util.hpp"
#include "../../include/cmdline.h"
#include <filesystem>

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
            sub_mat = darkChannelImg.submat(m, n, m + width - 1, n + height - 1);
            result.slice(i) = sub_mat;
            ++i;
        }
    }

    final = arma::min(result, 2);
    auto out = x_arma_to_cv_single(final);

    return out.clone();
}

double get_global_light(const cv::Mat &darkChannelImg) {
    //这里是简化的处理方式,A的最大值限定为220
    double minAtomsLight = 220;//经验值
    double maxValue = 0;
    cv::Point maxLoc;
    minMaxLoc(darkChannelImg, nullptr, &maxValue, nullptr, &maxLoc);
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

cv::Mat get_dehazed_img_guided_filter(const cv::Mat &src, const cv::Mat &darkChannelImg, double ratio) {
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
    fineTransmissionChannel[0] = fastGuidedFilter(srcChannel[0], transmissionChannel[0], 64, 0.01, 8);
    fineTransmissionChannel[1] = fastGuidedFilter(srcChannel[1], transmissionChannel[1], 64, 0.01, 8);
    fineTransmissionChannel[2] = fastGuidedFilter(srcChannel[2], transmissionChannel[2], 64, 0.01, 8);

    cv::split(dehazedImg, dehazedChannel);
    dehazedChannel[0] = get_dehazed_channel(srcChannel[0], fineTransmissionChannel[0], A);
    dehazedChannel[1] = get_dehazed_channel(srcChannel[1], fineTransmissionChannel[1], A);
    dehazedChannel[2] = get_dehazed_channel(srcChannel[2], fineTransmissionChannel[2], A);

    cv::merge(dehazedChannel, dehazedImg);

    return dehazedImg;
}

vips::VImage haze_remove(const vips::VImage &input, int radius, float ratio) {
    auto src = x_vips_to_cv_8u(input);
    auto darkChanelImg = dark_channel_image(src, radius);
    auto out = get_dehazed_img_guided_filter(src, darkChanelImg, ratio);
    auto dst = x_cv_to_vips_double(out);
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    return dst;
}

void copy_tfw_file(const char *in, const char *out) {
    std::fstream fin, fout;
    fin.open(in, std::ios_base::in);
    fout.open(out, std::ios_base::out | std::ios_base::trunc);
    if (!fin.is_open() || !fout.is_open()) {
        printf("open twf file error");
        exit(0);
    }

    char *buf = (char *) calloc(1, 1024);
    while (!fin.eof()) {
        fin.read(buf, 1024);
        fout.write(buf, fin.gcount());
    }
    //关闭文件，释放内存
    fin.close();
    fout.close();
    free(buf);
}

void save(vips::VImage &input, const char *filename) {
    input.write_to_file(filename, vips::VImage::option()->
            set("Q", 75));
}

int main(int argc, char **argv) {
    if (VIPS_INIT(argv[0])) {
        vips_error_exit("unable to init system environment.");
    }
    vips_cache_set_max(10);

    /* setup parameters:
     * Radius(int): 滤波半径大小 7
     * Radio(float): 薄雾比例 0.95
     * Input: 输入目录
     * Output: 输出目录
    */
    cmdline::parser theArgs;
    theArgs.add<std::string>("input", '\0', "input image directory", true);
    theArgs.add<std::string>("output", '\0', "output image directory", true);
    theArgs.add<int>("radius", '\0', "radius of the filter", false, 7);
    theArgs.add<float>("radio", '\0', "defog radio", false, 0.95);

    theArgs.parse_check(argc, argv);

    const char *input_dir = theArgs.get<std::string>("input").c_str();
    const char *output_dir = theArgs.get<std::string>("output").c_str();
    int radius = theArgs.get<int>("radius");
    float radio = theArgs.get<float>("radio");

    for (const auto &entry: std::filesystem::directory_iterator(input_dir)) {
        auto filename = entry.path().string(); // 文件名带目录
        size_t pos = filename.find('.', 0);
        auto suffix = filename.substr(pos + 1, 3);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);

        if ((suffix == "tif") || (suffix == "jpg")) {
            printf(">>>>> processing: \t%s\n", filename.c_str());
            auto img_data = vips::VImage::new_from_file(filename.c_str());
            auto haze_removed = haze_remove(img_data, radius, radio);

            auto output_name = std::string(output_dir).append("\\").append(entry.path().filename().string());
            save(haze_removed, output_name.c_str());

            auto tfw_file_in = filename;
            auto tfw_file_out = output_name;
            if (suffix == "tif") {
                tfw_file_in = tfw_file_in.replace(tfw_file_in.find("tif"), 3, "tfw");
                tfw_file_out = tfw_file_out.replace(tfw_file_out.find("tif"), 3, "tfw");
            } else {
                tfw_file_in = tfw_file_in.replace(tfw_file_in.find("jpg"), 3, "tfw");
                tfw_file_out = tfw_file_out.replace(tfw_file_out.find("jpg"), 3, "tfw");
            }

            if (std::filesystem::exists(std::filesystem::path(tfw_file_in))) {
                copy_tfw_file(tfw_file_in.c_str(), tfw_file_out.c_str());
            }
        }

    }
}