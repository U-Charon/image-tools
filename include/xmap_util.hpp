//
// Created by jerry on 2022/5/12.
//

#ifndef XMAPIMAGETOOLS_XMAP_UTIL_HPP
#define XMAPIMAGETOOLS_XMAP_UTIL_HPP

#include <vips/vips8>
#include <armadillo>
#include <opencv2/opencv.hpp>

/*
 * 获取当前系统时间（毫秒），除以1000就是秒
 */
void x_clear_image(VipsImage **image) {
    if (G_IS_OBJECT(*image)) g_clear_object(&image);
}

long long x_get_current_ms() {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );

    return ms.count();
}

arma::mat x_vips_to_arma_mat(const vips::VImage &in) {
    size_t size;
    auto vips_in = in.cast(VIPS_FORMAT_DOUBLE);
    void *data = vips_in.write_to_memory(&size);
    auto *data1 = (double *) data;

    arma::mat dst = arma::mat(data1, in.width(), in.height(), true);

    g_free(data);
    return dst;
}

/*
 * VIPS image转换为arma cube --- 有点慢，尽量避免使用
 * 应该用先转换为cv，然后从cv转换成arma cube，例如：
 *
 * auto cv = x_vips_to_cv_64f(vips);
 * auto cube = x_cv_to_arma_cube(cv);
 */
arma::cube x_vips_to_arma_cube(const vips::VImage &in) {
    size_t size;
    auto vips_in = in.cast(VIPS_FORMAT_DOUBLE);
    auto v = vips_in[2].bandjoin(vips_in[1]).bandjoin(vips_in[0]);

    arma::cube dst = arma::cube(in.width(), in.height(), 3);
    for (int i = 0; i < 3; i++) {
        void *data = v[i].write_to_memory(&size);
        auto *data1 = (double *) data;
        dst.slice(i) = arma::mat(data1, in.width(), in.height(), true);

        g_free(data);
    }

    return dst;
}

vips::VImage x_arma_mat_to_vips(const arma::mat &in) {

    return {vips_image_new_from_memory_copy(in.memptr(),
                                            sizeof(double) * in.n_elem,
                                            (int) in.n_rows,
                                            (int) in.n_cols,
                                            1,
                                            VIPS_FORMAT_DOUBLE
    )};
}

vips::VImage x_arma_cube_to_vips(const arma::cube &in) {

    vips::VImage r = {vips_image_new_from_memory_copy(in.slice(2).memptr(),
                                                      sizeof(double) * in.slice(2).n_elem,
                                                      (int) in.n_rows,
                                                      (int) in.n_cols,
                                                      1,
                                                      VIPS_FORMAT_DOUBLE
    )};
    vips::VImage g = {vips_image_new_from_memory_copy(in.slice(1).memptr(),
                                                      sizeof(double) * in.slice(1).n_elem,
                                                      (int) in.n_rows,
                                                      (int) in.n_cols,
                                                      1,
                                                      VIPS_FORMAT_DOUBLE
    )};
    vips::VImage b = {vips_image_new_from_memory_copy(in.slice(0).memptr(),
                                                      sizeof(double) * in.slice(0).n_elem,
                                                      (int) in.n_rows,
                                                      (int) in.n_cols,
                                                      1,
                                                      VIPS_FORMAT_DOUBLE
    )};

    return r.bandjoin(g).bandjoin(b);

}

/*
 * cv Mat到vipsVImage的转换，类型为double
 */
vips::VImage x_cv_to_vips_double(const cv::Mat &mat) {
    cv::Mat dst, src, result;

    if (mat.channels() == 1) {
        return {vips_image_new_from_memory_copy(mat.data,
                                                mat.elemSize() * mat.cols * mat.rows,
                                                mat.cols,
                                                mat.rows,
                                                mat.channels(),
                                                VIPS_FORMAT_DOUBLE
        )};
    } else {
        mat.convertTo(src, CV_32F);
        cv::cvtColor(src, result, cv::COLOR_BGR2RGB);
        result.convertTo(dst, CV_64F);

        return {vips_image_new_from_memory_copy(dst.data,
                                                dst.elemSize() * dst.cols * dst.rows,
                                                dst.cols,
                                                dst.rows,
                                                dst.channels(),
                                                VIPS_FORMAT_DOUBLE
        )};
    }
}

/*
 * VIPS到cv mat的转换，输出类型为CV_8UC3（0-255）3通道
 */
cv::Mat x_vips_to_cv_8u(const vips::VImage &in) {
    size_t dataSize;

    auto bgr = in[2].bandjoin(in[1]).bandjoin(in[0]);
    void *data = bgr.write_to_memory(&dataSize);
    cv::Mat mat = cv::Mat(bgr.height(), bgr.width(), CV_8UC3, data);

    return mat;
}

/*
 * VIPS到cv mat的转换，输出类型为CV_8UC1（0-255）1通道
 */
cv::Mat x_vips_to_cv_single(const vips::VImage &in) {
    size_t dataSize;

    auto in1 = in.cast(VIPS_FORMAT_DOUBLE);
    void *data = in1.write_to_memory(&dataSize);
    cv::Mat mat = cv::Mat(in1.height(), in1.width(), CV_64FC1, data);

    return mat;
}

/*
 * VIPS到cv mat的转换，输出类型为CV_64FC3（double）3通道
 */
cv::Mat x_vips_to_cv_64f(const vips::VImage &in) {
    size_t dataSize;

    auto bgr = in[2].bandjoin(in[1]).bandjoin(in[0]);
    bgr = bgr.cast(VIPS_FORMAT_DOUBLE);
    void *data = bgr.write_to_memory(&dataSize);
    cv::Mat mat = cv::Mat(bgr.height(), bgr.width(), CV_64FC3, data);
    return mat;
}

/*
 * cv mat 3通道 到 arma Cube转换，类型为模板类型
 */
template<typename T, int NC>
arma::Cube<T> x_cv_to_arma_cube(const cv::Mat_<cv::Vec<T, NC>> &src) {
    std::vector<cv::Mat_<T>> channels;
    arma::Cube<T> dst(src.cols, src.rows, NC);
    for (int c = 0; c < NC; ++c)
        channels.emplace_back(src.rows, src.cols, dst.slice(c).memptr());
    cv::split(src, channels);
    return dst;
}

/*
 * cv mat 单通道到 arma Mat转换，类型为模板类型
 */
template<typename T>
// https://stackoverflow.com/questions/26118862/convert-cvmat-to-armamat
arma::Mat<T> x_cv_to_arma_mat(const cv::Mat_<T> &src) {
    cv::Mat_<T> temp(src);
    arma::Mat<T> dst = arma::Mat<T>(temp.template ptr<T>(), temp.cols, temp.rows, true, true);
    return dst;
}

/*
 * arma Cube 3通道到 cv Mat转换，类型为模板类型
 */
template<typename T>
cv::Mat x_arma_to_cv(const arma::Cube<T> &src) {
    std::vector<cv::Mat_<T>> channels;
    for (size_t c = 0; c < src.n_slices; ++c) {
        auto *data = const_cast<T *>(src.slice(c).memptr());
        channels.emplace_back(int(src.n_cols), int(src.n_rows), data);
    }
    cv::Mat dst;
    cv::merge(channels, dst);
    return dst;
}

/*
 * arma Mat 单通道到 cv Mat转换，类型为模板类型
 */
template<typename T>
cv::Mat x_arma_to_cv_single(const arma::Mat<T> &src) {
    return {int(src.n_cols), int(src.n_rows), CV_8UC1, const_cast<T *>(src.memptr())};
}

/*
 * 显示cv image，可以试单通道也可以是3通道
 */
void x_display_cv_image(const cv::Mat &in, const std::string &name, const int win_num) {
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, in);
    cv::moveWindow(name, (win_num * in.cols + 10), 20);
}

/*
 * 显示VIPS image，可以试单通道也可以是3通道
 * 2022-10-20：
 *      修改加入4波段VIPS image的显示
 */
void x_display_vips_image(const vips::VImage &in, const std::string &name, const int win_num) {
    size_t dataSize;
    cv::Mat mat;
    void *data = in.write_to_memory(&dataSize);

    if (in.bands() == 3) {
        mat = cv::Mat(in.height(), in.width(), CV_8UC3, data);
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    } else if (in.bands() == 4) {
        mat = cv::Mat(in.height(), in.width(), CV_8UC4, data);
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    } else {
        mat = cv::Mat(in.height(), in.width(), CV_8UC1, data);
    }
    cv::namedWindow(name, cv::WINDOW_FREERATIO); // 窗口大小自适应比例
    cv::imshow(name, mat);
    cv::moveWindow(name, (win_num * in.width() + 10), 20);
}

/*
 * 获取arma Cube<T> 的均值、最小、最大值
 * Return : vector[0] = mean
 *          vector[1] = min
 *          vector[2] = max
 */
std::vector<double> x_arma_cube_stats(const arma::cube &in) {
    std::vector<double> result;
    arma::mat r, g, b;

    r = in.slice(0);
    g = in.slice(1);
    b = in.slice(2);

    double mean_slice0 = arma::mean(arma::mean(r));
    double mean_slice1 = arma::mean(arma::mean(g));
    double mean_slice2 = arma::mean(arma::mean(b));
    auto mean = (mean_slice0 + mean_slice1 + mean_slice2) / 3;
    result.push_back(mean);

    double min_slice0 = arma::min(arma::min(r));
    double min_slice1 = arma::min(arma::min(g));
    double min_slice2 = arma::min(arma::min(b));
    auto min = std::min(std::min(min_slice0, min_slice1), min_slice2);
    result.push_back(min);

    double max_slice0 = arma::max(arma::max(r));
    double max_slice1 = arma::max(arma::max(g));
    double max_slice2 = arma::max(arma::max(b));
    auto max = std::max(std::max(max_slice0, max_slice1), max_slice2);
    result.push_back(max);

    return result;
}

/*
 * Return arma cube mean
 */
double x_arma_cube_mean(const arma::cube &in) {
    arma::mat r, g, b;

    r = in.slice(0);
    g = in.slice(1);
    b = in.slice(2);

    return (arma::mean(arma::mean(r)) + arma::mean(arma::mean(g)) + arma::mean(arma::mean(b))) / 3.;
}

/*
 * Return arma cube std
 */
double x_arma_cube_std(const arma::cube &in) {
    arma::vec v = arma::vectorise(in);
    return arma::stddev(v);
}

/*
 * cv padding, 缺省使用cv::BORDER_REFLECT镜像模式
 * Input: vips::VImage
 *        int: top, bottom, left, right
 * Output: cv:Mat
 */
cv::Mat x_cv_padding(const vips::VImage &in, int top, int bottom, int left, int right) {
    auto src = x_vips_to_cv_64f(in);
    cv::Mat padded;

    cv::copyMakeBorder(src, padded, top, bottom, left, right,
                       cv::BORDER_REFLECT);
//    padded.convertTo(padded, CV_64F);

    return padded;
}

/*
 * VIPS 版本的 cv中countNonZero 函数
 */
int x_count_non_zero(const vips::VImage &in) {
    auto img_hist = in.hist_find();
    auto total_count = in.width() * in.height();
    auto zero_count = int(img_hist(0, 0)[0]);

    return total_count - zero_count;
}

/*
 * 统计image中的非零并且非65535的像素个数
 */
double x_count_non_zero_and_white(const vips::VImage &in) {
    double white_count = 0.0;
    double total_count = 0.0;
    double zero_count = 0.0;

    auto img_hist = in.hist_find();
    total_count = (double) in.width() * in.height();
    zero_count = (double) (img_hist(0, 0)[0]);
    if (img_hist.width() == 65536) {
        white_count = (double) (img_hist(65535, 0)[0]);
    }
    printf("total=%f, zero=%f, white=%f\n", total_count, zero_count, white_count);
    return total_count - zero_count - white_count;
}

/*
 * image padding [top, bottom, left, right]
 */
vips::VImage x_vips_padding(const vips::VImage &in, int top, int bottom, int left, int right) {
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

#endif //XMAPIMAGETOOLS_XMAP_UTIL_HPP
