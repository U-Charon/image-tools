//
// Created by jerry on 2022/4/11.
//

/*
 * 2022-07-22：反馈bug，方法里calculate_hist_total，统计像素的时候应该排除掉0像素
 *      添加一个函数x_count_non_zero，统计3波段影像非零像素个数;
 *      ## 未来有可能还会去掉255像素值
 *
 * 2022-07-23: 关联发现bug，应该去掉黑色像素的个数，之前没有，导致输出影像会偏暗。
 *      在方法optimized_stretch中：
 *      定义一个新变量：total_non_zero_pixel
 *      注释掉了以下两行，改变成现在的：
 *      auto a = find_low(in, min_percent * in.height() * in.width());
 *      auto b = find_high(in, (1 - max_percent) * in.height() * in.width());
 *
 * 2022-07-23: 同步发现 find_low, find_high 方法中bug:
 *      直接设置i为1，j为w-1就行了，因为：
 *      1、统计已经把为0的像素个数排查掉了，直方图直接从1开始就行了，hist(0,0)[0]不参与统计，跟0的个数有无都无关了；
 *      2、高位还是用j=w-1，暂时不考虑255和65535情况
 *      3、calculate_hist_total 方法中也同步去掉i,j的判断
 *
 * 2022-07-24: 检测发现sigmoid拉伸也有同样问题，需要在sigmoid_stretch和sigmoid_stretch_batch中修改：
 *      在方法sigmoid_stretch中：
 *      1、定义一个新变量：total_non_zero_pixel，注释掉了以下两行，改变成现在的：
 *      auto a = find_low(in, min_percent * in.height() * in.width());
 *      auto b = find_high(in, (1 - max_percent) * in.height() * in.width());
 *
 * 2022-07-24:
 *      调整4个拉伸方法（2个拉伸，2个batch）方法的返回值，让函数变成void, 返回的图像在golang中定义VipsImage *out来接收，
 *      这样尝试在dll返回后没有任何遗留对象需要释放，尝试解决一直以来有的内存泄漏问题
 *
 * 2022-08-01:
 *      调整calculate_hist_total和find_high方法，加入了去除65535纯白像素部分，这两个函数一个对应批处理拉伸的统计，
 *      一个对应单图像拉伸统计。
 *      optimized_stretch : 加入排除65535的部分
 *
 * 2022-08-02:
 *      济南反馈，optimized_stretch拉伸的最大值要設成255，不能是254，修改max_value参数
 *
 * 2022-08-15：
 *      根据济南需求，加入jpg的优化拉伸:
 *          1. 修改OptimizedStretchBatch方法，加入jpg部分
 *          2. 修改SigmoidStretchBatch方法，加入jpg部分
 *          3. CalculateHistTotal方法加入jpg的判断
 *
 * 2022-08-15：
 *      save_tiff，加入压缩，使用VIPS_FOREIGN_TIFF_COMPRESSION_JPEG，Q为75
 *
 * 2022-10-18：
 *      济南反馈拉伸和S拉伸算法的批处理部分有内存泄漏，跑大量影像会崩溃。本地测试确实在go里每次通过CGO返回VipsImage的指针变量，
 *      然后在go里保存以后，内存不会释放（尝试过image.close()，但是无效）--> 现改为批处理每次调用CGO时只传入待处理影像的路径，
 *      算法调用和保存在C++里完成。
 *
 * 2022-10-19:
 *      大影像统计非零像素值用int64会超界，现在在xmap_util.hpp中的x_count_non_zero_and_white方法全部改为double返回
 *      c++中double型的最大值和最小值：
 *      1、负值取值范围为-1.79769313486231570E+308到-4.94065645841246544E-324；
 *      2、正值取值范围为4.94065645841246544E-324到1.79769313486231570E+308
 *
 * 2022-10-25:
 *      植被增强里需判断影像有无黑的部分（纯0部分），要记录下来贴回去
 *
 * 2022-11-03:
 *      根据济南需求提供修改后的色迁算法，只有一张小的底图和一堆原始图做色迁
 *
 *
 * 2022-11-07:
 *      济南反馈的直方图匹配算法有像素化问题，经查找发现是VIPS库在累积直方图上调用hist_norm的时候，会自动变回UCHAR，损失了精度；
 *      就导致了后面的hist_match调用会有很多像素映射不上变黑了。
 *      根据下面的链接做了调整，自己实现了LUT来调用
 *      https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/
 *
 *
 */

#include "xmap.h"
#include "../../include/xmap_util.hpp"
#include <filesystem>

// 图像边界处理
cv::Mat image_make_border(const cv::Mat &src) {
    int w = cv::getOptimalDFTSize(src.cols); // 获取DFT变换的最佳宽度
    int h = cv::getOptimalDFTSize(src.rows); // 获取DFT变换的最佳高度
//    printf("inside make border w=%d, h=%d\n", w, h);

    cv::Mat padded;
    // 常量法扩充图像边界，常量 = 0
//    cv::copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols,
//                       cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols,
                       cv::BORDER_REFLECT101);
//    padded.convertTo(padded, CV_64FC1);

    return padded;
}

// 高斯低通滤波核函数
cv::Mat gaussian_low_pass_kernel(int rows, int cols, double sigma) {
//    cv::Mat gaussianBlur(rows, cols, CV_32FC1); //，CV_32FC1
    cv::Mat gaussianBlur(rows, cols, CV_64FC1); //，CV_32FC1
    double d0 = sigma;//高斯函数参数，越小，频率高斯滤波器越窄，滤除高频成分越多，图像就越平滑

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double d = pow((double(abs(i - rows / 2) - (rows - 1.0) / 2.0)), 2.0) +
                       pow((double(abs(j - cols / 2) - (cols - 1.0) / 2.0)), 2.0);
            gaussianBlur.at<double>(i, j) = exp(-d / (2 * d0 * d0));
        }
    }

    return gaussianBlur;
}

int find_low(const vips::VImage &in, double T) {
    auto hist = in.hist_find();

    double min_total = 0;
    int i = 1;
    while (min_total <= T) {
        min_total += hist(i, 0)[0];
        i += 1;
    }
    auto low = i - 1;

    return low;
}

int find_high(const vips::VImage &in, double T) {
    auto hist = in.hist_find();
    auto w = hist.width();
    int j;

    if (w == 65536) {
        j = w - 2;
    } else { j = w - 1; }   // 如果hist的宽度是65536，说明有白色要去掉，w-1就是65535的个数，w-2就是到65534，相当于白色剔除掉；

    double max_total = 0;
    while (max_total <= T) {
        max_total += hist(j, 0)[0];
        j -= 1;
    }
    auto high = j + 1;

    return high;
}

VipsImage *new_image_from_file(const char *filename) {
//    return vips_image_new_from_file(filename, NULL);
    VipsImage *image;
    if (!(image = vips_image_new_from_file(filename,
                                           "access", VIPS_ACCESS_SEQUENTIAL,
                                           NULL)))
        vips_error_exit(nullptr);

    return image;
}

void new_image_from_file1(const char *filename, VipsImage **out) {

    if (!(*out = vips_image_new_from_file(filename,
//                                          "access", VIPS_ACCESS_SEQUENTIAL,
                                          NULL)))
        vips_error_exit(nullptr);

}

void coarse_light_remove(VipsImage *input, VipsImage **output, double offset, double sigmaD0) {
    printf("entering simple light removal...\n");
    auto start = x_get_current_ms();
//    VipsImage *output;
//    vips::VImage in, src;
//
//    in = vips::VImage(input, vips::NOSTEAL) / 255;

    vips::VImage in, in1, src, dst;
//    size_t data_size;
//    void *buff;
//    in = (vips::VImage) input;
    in1 = vips::VImage(input, vips::NOSTEAL);

//    auto input_copy = in1.copy();
//    auto input_zero = (input_copy == 0).ifthenelse(0, 1);
    auto input_zero = (in1 == 0).ifthenelse(0, 1);
//
//    buff = in1.write_to_memory(&data_size);
//    in = vips::VImage::new_from_memory(buff, data_size, in1.width(), in1.height(),
//                                             in1.bands(), VIPS_FORMAT_UCHAR);
//    in = in / 255.;

    in = in1 / 255.;

    auto w = cv::getOptimalDFTSize(in.width());   // cols
    auto h = cv::getOptimalDFTSize(in.height());  // rows

    auto padded = image_make_border(x_vips_to_cv_64f(in));
    src = x_cv_to_vips_double(padded);
    auto lowKernel = gaussian_low_pass_kernel(h, w, sigmaD0);
    auto vipKernel = x_cv_to_vips_double(lowKernel);
    auto rResult = src[0] - src[0].fwfft().freqmult(vipKernel) + src[0].avg() + offset;
    auto gResult = src[1] - src[1].fwfft().freqmult(vipKernel) + src[1].avg() + offset;
    auto bResult = src[2] - src[2].fwfft().freqmult(vipKernel) + src[2].avg() + offset;

    dst = rResult.bandjoin(gResult).bandjoin(bResult);
    dst = dst.extract_area(0, 0, in.width(), in.height());

    dst = dst * input_zero;
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &(*output), NULL);

    padded.release();
    lowKernel.release();

    auto end = x_get_current_ms();
    printf("simple light removal time: %.8f second.\n", (double) (end - start) / 1000);
    printf("done. simple light removal...\n");

}

vips::VImage ones(int rows, int cols) {
    return (vips::VImage::black(rows, cols) + 1.0);
}

vips::VImage convolve_vips(const vips::VImage &src, const vips::VImage &kernel) {
    vips::VImage conv_out = vips::VImage::black(src.width(), src.height());
    int radius = (int) (((kernel.width() - 1) / 2) - 1) / 2;
    auto src1 = src.cast(VIPS_FORMAT_DOUBLE);

    for (int b = 0; b < 3; b++) {
        conv_out += src1[b].conv(kernel, vips::VImage::option()->
                set("layers", radius)->
                set("cluster", radius));
    }

    return conv_out;
}

VipsImage *pixel_balance(VipsImage *input, int blk_size) {
    printf("entering pixel balance...\n");
    auto start = x_get_current_ms();

    VipsImage *output;
    vips::VImage in, in1, in2;
    int cut = (blk_size - 1) / 2;
    int radius = floor((double) (cut - 1) / 2 * 0.1); // 取kernel的半径的10%
    size_t data_size;
    void *buff;

//    in1 = vips::VImage(input, vips::NOSTEAL);
//    auto h = in1.height();
//    auto w = in1.width();
//    buff = in1.write_to_memory(&data_size);
//    in2 = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
//    in = in2.cast(VIPS_FORMAT_DOUBLE) / 255;

    in = vips::VImage(input, vips::NOSTEAL) / 255;
    in = in.cast(VIPS_FORMAT_DOUBLE);
    auto h = in.height();
    auto w = in.width();

    auto deta_x = (in.extract_area(1, 0, w - 1, h)) -
                  (in.extract_area(0, 0, w - 1, h));         // [h, w-1, 3]
    auto deta_y = (in.extract_area(0, 1, w, h - 1)) -
                  (in.extract_area(0, 0, w, h - 1));         // [h-1, w, 3]

    auto deta_x_square = deta_x.extract_area(0, 1, deta_x.width(), deta_x.height() - 1).pow(2.0);
    auto deta_y_square = deta_y.extract_area(1, 0, deta_y.width() - 1, deta_y.height()).pow(2.0);
    auto residual = ((deta_x_square + deta_y_square) / 2).pow(0.5);

    auto def_filter = ones((blk_size - 1), (blk_size - 1)) / pow((blk_size - 1), 2.0);
    auto nbr_filter = ones(blk_size, blk_size) / pow(blk_size, 2.0);

    auto def_v_tmp = convolve_vips(residual, def_filter);
    auto mean_nbr = in.conv(nbr_filter,
                            vips::VImage::option()->
                                    set("layers", radius)->
                                    set("cluster", radius));
    auto std_nbr = (in - mean_nbr).pow(2.0).conv(nbr_filter,
                                                 vips::VImage::option()->
                                                         set("layers", radius)->
                                                         set("cluster", radius)).pow(0.5);
    auto definition_value = def_v_tmp.extract_area(cut, cut,
                                                   residual.width() - cut,
                                                   residual.height() - cut);

    // 找到definition value 的最大像素坐标
    auto def_value_stat = definition_value.stats();

    auto x_cord = def_value_stat(8, 0)[0];
    auto y_cord = def_value_stat(9, 0)[0];
    auto start_x = x_cord - cut;
    auto start_y = y_cord - cut;
    vips::VImage ref_block;

    if (start_x > 0 && start_y > 0) {
        ref_block = in.extract_area((int) start_x, (int) start_y, blk_size, blk_size);
    } else {
        ref_block = in.extract_area((int) x_cord, (int) y_cord, blk_size, blk_size);
    }

    // 计算weights
    std::vector<double> mean_ref, std_ref;
    for (int i = 0; i < 3; i++) {
        mean_ref.push_back(ref_block[i].avg());
        std_ref.push_back(ref_block[i].deviate());
    }

    auto w_s = std_ref / (std_ref + std_nbr);
    auto w_m = mean_ref / (mean_ref + mean_nbr);
    auto alpha = w_s * std_ref / (w_s * std_nbr + (1.0 - w_s) * std_ref);
    auto beta = w_m * mean_ref + (1.0 - w_m - alpha) * mean_nbr;

//    vips::VImage result = vips::VImage::black(w, h);
    auto result = alpha * in + beta;
    result = (result * 255).cast(VIPS_FORMAT_UCHAR);
//    auto end = x_get_current_ms();
//    printf("pixel balance computation time: %.8f second.\n", (double) (end - start) / 1000);
//    printf("done. pixel balance...\n");

    buff = result.write_to_memory(&data_size);
    auto dst = vips::VImage::new_from_memory_steal(buff, data_size, w, h,
                                                   result.bands(), VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);

//    g_object_ref(output);

    auto end = x_get_current_ms();
    printf("total computation time: %.8f second.\n", (double) (end - start) / 1000);
    printf("done. pixel balance...\n");

    return output;

}

arma::mat convolve_vips_mat(const arma::cube &src, const vips::VImage &kernel) {
    int kernel_param = (int) (kernel.width() - 1) / 2;
    auto vips_src = x_arma_cube_to_vips(src);
    vips::VImage total = vips::VImage::black(vips_src.width(), vips_src.height());

    for (int i = 0; i < 3; i++) {
        total += vips_src[i].conv(kernel, vips::VImage::option()->
                set("layers", kernel_param)->
                set("cluster", kernel_param));
    }

    auto arma_conv = x_vips_to_arma_mat(total);

    return arma_conv;
}

arma::cube convolve_vips_cube(const arma::cube &src, const vips::VImage &kernel) {
    int kernel_param = (int) (kernel.width() - 1) / 2;
    auto vips_src = x_arma_cube_to_vips(src);

    auto vips_conv = vips_src.conv(kernel, vips::VImage::option()->
            set("layers", kernel_param)->
            set("cluster", kernel_param));

    auto cv_conv = x_vips_to_cv_64f(vips_conv);
    auto arma_conv = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) cv_conv);

    return arma_conv;
}

VipsImage *pixel_balance1(VipsImage *input, int blk_size) {
    using namespace arma;
    VipsImage *output;

    printf("entering pixel balance1...\n");
    auto start = x_get_current_ms();

    auto in = vips::VImage(input, vips::NOSTEAL) / 255;
    in = in.cast(VIPS_FORMAT_DOUBLE);
    auto h = in.height();
    auto w = in.width();
    int cut = (blk_size - 1) / 2;

    auto arma_in = x_vips_to_arma_cube(in);

    cube deta_x = arma_in.tube(1, 0, w - 1, h - 1) -
                  arma_in.tube(0, 0, w - 2, h - 1); // [row x col] 475x476
    cube deta_y = arma_in.tube(0, 1, w - 1, h - 1) -
                  arma_in.tube(0, 0, w - 1, h - 2); // [row x col] 476x475


    cube deta_x_square = pow(deta_x.tube(0, 1, deta_x.n_rows - 1, deta_x.n_cols - 1), 2.0);
    cube deta_y_square = pow(deta_y.tube(1, 0, deta_y.n_rows - 1, deta_y.n_cols - 1), 2.0);
    cube residual = pow(((deta_x_square + deta_y_square) / 2), 0.5);

    mat def_filter = arma::ones((blk_size - 1), (blk_size - 1)) / pow((blk_size - 1), 2.0);
    mat nbr_filter = arma::ones(blk_size, blk_size) / pow(blk_size, 2.0);

    mat definition_value = zeros((w - 1) - (blk_size - 1), (h - 1) - (blk_size - 1));
    cube mean_nbr = zeros(w, h, 3);
    cube std_nbr = zeros(w, h, 3);

    mat def_temp(residual.n_rows, residual.n_cols);


    auto def_filter_vips = ones((blk_size - 1), (blk_size - 1)) / pow((blk_size - 1), 2.0);
    auto nbr_filter_vips = ones(blk_size, blk_size) / pow(blk_size, 2.0);
    def_temp = convolve_vips_mat(residual, def_filter_vips);

    mean_nbr = convolve_vips_cube(arma_in, nbr_filter_vips);
    std_nbr = pow(convolve_vips_cube(pow((arma_in - mean_nbr), 2.0), nbr_filter_vips), 0.5);

    definition_value = def_temp.submat(cut, cut, residual.n_rows - cut, residual.n_cols - cut);

    rowvec b = max(definition_value, 0);
    colvec c = max(definition_value, 1);
    int y = (int) index_max(b);
    int x = (int) index_max(c);

    cube ref_block(blk_size, blk_size, 3);
    if (((x - cut) > 0) && ((y - cut) > 0)) {
        ref_block = arma_in.tube(x - cut, y - cut, size(blk_size, blk_size));
    } else {
        ref_block = arma_in.tube(x, y, size(blk_size, blk_size));
    }

    colvec mean_ref = mean(mean(ref_block, 0), 0);
    cube std_cube = reshape(ref_block, blk_size * blk_size, 3, 1);
    rowvec std_ref = stddev(std_cube.slice(0));

    cube w_s(w, h, 3),
            w_m(w, h, 3), alpha(w, h, 3),
            beta(w, h, 3);

    for (int i = 0; i < 3; i++) {
        w_s.slice(i) = std_ref(i) / (std_ref(i) + std_nbr.slice(i));
        w_m.slice(i) = mean_ref(i) / (mean_ref(i) + mean_nbr.slice(i));

        alpha.slice(i) = w_s.slice(i) * std_ref(i) / (w_s.slice(i) % std_nbr.slice(i) + (1.0 - w_s.slice(i)) * std_ref(i));
        beta.slice(i) = w_m.slice(i) * mean_ref(i) + (1.0 - w_m.slice(i) - alpha.slice(i)) % mean_nbr.slice(i);
    }

    cube arma_dst = alpha % arma_in + beta;
    auto dst = x_arma_cube_to_vips(arma_dst);
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);
    auto end = x_get_current_ms();
    printf("pixel balance1 total time %.8f second.\n", (double) (end - start) / 1000);

    return output;
}

std::vector<int> find_low_high(const vips::VImage &in, double T) {
    auto hist = in.hist_find();
    auto w = hist.width();

    double min_total = 0, max_total = 0;
    int i = 0, j = w - 1;
    while (min_total <= T) {
        min_total += hist(i, 0)[0];
        i += 1;
    }
    auto low = i - 1;

    while (max_total <= T) {
        max_total += hist(j, 0)[0];
        j -= 1;
    }
    auto high = j + 1;

    return std::vector<int>{low, high};
}

void save(vips::VImage &input, const char *filename) {
    input.write_to_file(filename, vips::VImage::option()->
            set("compression", VIPS_FOREIGN_TIFF_COMPRESSION_NONE)->
            set("Q", 100));
}

VipsImage *radiation_correction(VipsImage *input, double min_value, double max_value, double t1, double gamma, double lambda_adjust) {
    printf("entering keep mean stretch...\n");
    auto start = x_get_current_ms();
    VipsImage *output;
    vips::VImage in, in1, in2, band, result[3];
    double min_, max_;
    double lambda, T;
    size_t data_size;
    void *buff;

    in = vips::VImage(input, vips::NOSTEAL);
//    in1 = vips::VImage(input, vips::NOSTEAL);
//    auto h = in1.height();
//    auto w = in1.width();
//    buff = in1.write_to_memory(&data_size);
//    in2 = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
//    in = in2.cast(VIPS_FORMAT_DOUBLE) / 255;

//    min_value = min_value ;
//    max_value = max_value ;
//    in = (vips::VImage) input / 255;
    T = t1 * in.width() * in.height();

    for (int i = 0; i < 3; i++) {
        band = in.extract_band(i);

//        auto maxv_band = band.hist_find().stats()(8, 0)[0];
        auto maxv_band = band.hist_find().stats()(8, 0)[0];
        auto meanv_band = band.avg();
        auto low_high = find_low_high(band, T);
        min_ = low_high[0];
        max_ = low_high[1];

//        printf("maxv=%d, meanv=%f, low=%f, high=%f\n", maxv_band, meanv_band, min_, max_);
        band = (band < min_).ifthenelse(min_value, band); // < low 设为min value
        band = (band >= max_).ifthenelse(max_value, band);

        auto v1 = min_value + (band - min_) * ((meanv_band - min_value) / (meanv_band - min_));
        band = (band >= min_ & band < meanv_band).ifthenelse(v1, band);

        auto v2 = meanv_band + (band - meanv_band) * ((max_value - meanv_band) / (max_ - meanv_band));
        band = (band >= meanv_band & band < max_).ifthenelse(v2, band);

        if (maxv_band < 128) {
            lambda = pow((maxv_band / 128), gamma);
        } else {
            lambda = pow((128 / maxv_band), gamma);
        }

        result[i] = band.pow(lambda + lambda_adjust);
        result[i] = (result[i] > max_value).ifthenelse(max_value, result[i]);

    }
    auto dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);
    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
    printf("done. keep mean stretch...\n");

    return output;
}

/*
 * order = 1 : BGR
 * order = 0 : RGB
 */
void optimized_stretch(VipsImage *input, VipsImage **output, double min_percent, double max_percent,
                       double min_adjust, double max_adjust, int order) {
    printf("entering optimize stretch...\n");
    auto start = x_get_current_ms();
//    VipsImage *output;
    vips::VImage in, in1, band, dst;

    in = vips::VImage(input, vips::NOSTEAL);

    int channels = in.bands();
    vips::VImage result[channels];
    if (order == 1) {
        if (channels == 3) {
            in = in[2].bandjoin(in[1]).bandjoin(in[0]);
        } else if (channels == 4) {
            in = in[2].bandjoin(in[1]).bandjoin(in[0]).bandjoin(in[3]);
        }
    }

//    double c, d, min_value = 1 / 255., max_value = 254 / 255.;
    double c, d, min_value = 1 / 255., max_value = 1.0;
    auto input_copy = in.copy();
    auto input_zero = (input_copy == 0).ifthenelse(0, 1);
//    auto total_non_zero_pixel = x_count_non_zero(in);
    auto total_non_zero_pixel = x_count_non_zero_and_white(in);

    if (channels == 1) {

        auto a = find_low(in, min_percent * total_non_zero_pixel);
        auto b = find_high(in, (1 - max_percent) * total_non_zero_pixel);
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        in = (in < c).ifthenelse(min_value, in);
        auto v1 = (in - c) / (d - c) * (max_value - min_value) + min_value;
        in = (in >= c & in <= d).ifthenelse(v1, in);

        dst = (in > d).ifthenelse(max_value, in);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
        vips_copy(dst.get_image(), *(&output), NULL);
    } else {
        for (int i = 0; i < channels; i++) {
            band = in.extract_band(i);
            auto a = find_low(band, min_percent * total_non_zero_pixel);
            auto b = find_high(band, (1 - max_percent) * total_non_zero_pixel);
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min_value, band);
            auto v1 = (band - c) / (d - c) * (max_value - min_value) + min_value;
            band = ((band >= c) & (band <= d)).ifthenelse(v1, band);
            result[i] = (band > d).ifthenelse(max_value, band);
        }
        if (channels == 3) {
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            vips_addalpha(dst.get_image(), *(&output), NULL);
        } else {
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            vips_copy(dst.get_image(), *(&output), NULL);
        }
    }

    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
    printf("done. optimize stretch...\n");
}

void optimized_stretch_batch_calculate(char *input_name, char *output_name, double min_adjust, double max_adjust,
                                       int order, const int *low, const int *high) {
    printf("entering optimize stretch batch with calculate process...\n");
    auto start = x_get_current_ms();

    vips::VImage band, dst;
    auto input = vips::VImage::new_from_file(input_name);
    auto channels = input.bands();
    vips::VImage result[channels];

    if (order == 1) {
        if (channels == 3) {
            input = input[2].bandjoin(input[1]).bandjoin(input[0]);
        } else if (channels == 4) {
            input = input[2].bandjoin(input[1]).bandjoin(input[0]).bandjoin(input[3]);
        }
    }

//    double a, b, c, d, min_value = 1 / 255., max_value = 254 / 255.;
    double a, b, c, d, min_value = 1 / 255., max_value = 1.0;
    auto input_zero = (input == 0).ifthenelse(0, 1);

    if (channels == 1) {
        a = low[0];
        b = high[0];
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        input = (input < c).ifthenelse(min_value, input);
        auto v1 = (input - c) / (d - c) * (max_value - min_value) + min_value;
        input = (input >= c & input <= d).ifthenelse(v1, input);

        dst = ((input > d) & (input < 65535)).ifthenelse(max_value, input);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
    } else {
        for (int i = 0; i < channels; i++) {
            band = input.extract_band(i);
            a = low[i];
            b = high[i];
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min_value, band);
            auto v1 = (band - c) / (d - c) * (max_value - min_value) + min_value;
            band = ((band >= c) & (band <= d)).ifthenelse(v1, band);
            result[i] = (band > d).ifthenelse(max_value, band);
        }
    }

    switch (channels) {
        case 3:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        case 4:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        default:
            break;
    }

    save(dst, output_name);

    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
    printf("done. optimize batch stretch with calculate...\n");
}

void optimized_stretch_batch_no_calculate(char *input_name, char *output_name, double min_percent, double max_percent,
                                          double min_adjust, double max_adjust, int order) {
    printf("entering optimize stretch batch NO calculate process...\n");
    auto start = x_get_current_ms();

    vips::VImage band, dst;
    auto input = vips::VImage::new_from_file(input_name);
    auto channels = input.bands();
    vips::VImage result[channels];


    if (order == 1) {
        if (channels == 3) {
            input = input[2].bandjoin(input[1]).bandjoin(input[0]);
        } else if (channels == 4) {
            input = input[2].bandjoin(input[1]).bandjoin(input[0]).bandjoin(input[3]);
        }
    }

//    double a, b, c, d, min_value = 1 / 255., max_value = 254 / 255.;
    double c, d, min_value = 1 / 255., max_value = 1.0;
    auto input_zero = (input == 0).ifthenelse(0, 1);
    auto total_non_zero_pixel = x_count_non_zero_and_white(input);

    if (channels == 1) {
        auto a = find_low(input, min_percent * total_non_zero_pixel);
        auto b = find_high(input, (1 - max_percent) * total_non_zero_pixel);
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        input = (input < c).ifthenelse(min_value, input);
        auto v1 = (input - c) / (d - c) * (max_value - min_value) + min_value;
        input = (input >= c & input <= d).ifthenelse(v1, input);

        dst = ((input > d) & (input < 65535)).ifthenelse(max_value, input);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
    } else {
        for (int i = 0; i < channels; i++) {
            band = input.extract_band(i);
            auto a = find_low(band, min_percent * total_non_zero_pixel);
            auto b = find_high(band, (1 - max_percent) * total_non_zero_pixel);
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min_value, band);
            auto v1 = (band - c) / (d - c) * (max_value - min_value) + min_value;
            band = ((band >= c) & (band <= d)).ifthenelse(v1, band);
            result[i] = (band > d).ifthenelse(max_value, band);
        }
    }

    switch (channels) {
        case 3:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        case 4:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        default:
            break;
    }

    save(dst, output_name);

    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
    printf("done. optimize batch stretch NO calculate...\n");
}

void sigmoid_stretch(VipsImage *input, VipsImage **output, double min, double max, double min_percent,
                     double max_percent, double min_adjust,
                     double max_adjust, double alpha, double beta, int order) {
    printf("entering sigmoid stretch...\n");
    auto start = x_get_current_ms();
//    VipsImage *output;
    vips::VImage in, band, dst;

    double c, d;
    min = min / 255.;
    max = max / 255.;

    in = vips::VImage(input, vips::NOSTEAL);
    int channels = in.bands();
    vips::VImage result[channels];

    if (order == 1) {
        if (channels == 3) {
            in = in[2].bandjoin(in[1]).bandjoin(in[0]);
        } else if (channels == 4) {
            in = in[2].bandjoin(in[1]).bandjoin(in[0]).bandjoin(in[3]);
        }
    }

    auto input_zero = (in == 0).ifthenelse(0, 1);
    auto total_non_zero_pixel = x_count_non_zero(in);
    if (channels == 1) {
//        a = find_low(in, min_percent * in.height() * in.width());
//        b = find_high(in, (1 - max_percent) * in.height() * in.width());
        auto a = find_low(in, min_percent * total_non_zero_pixel);
        auto b = find_high(in, (1 - max_percent) * total_non_zero_pixel);
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        in = (in < c).ifthenelse(min, in);
        auto v1 = (in - c) / (d - c) * (max - min) + min;
        in = (in >= c & in <= d).ifthenelse(v1, in);
        dst = (in > d).ifthenelse(max, in);
        // ---------------------------------- sigmoid contrast ---------------------------------------------//
        auto temp = (dst > 0).ifthenelse(dst, alpha);
        auto x = (beta * (alpha - temp)).exp();
        x = (x == 1).ifthenelse(0, x);
        dst = 1 / (1 + x);
        dst = (dst == 1).ifthenelse(0, dst);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
        vips_copy(dst.get_image(), *(&output), NULL);
    } else {
        for (int i = 0; i < channels; i++) {
            band = in.extract_band(i);
//            a = find_low(band, min_percent * band.height() * band.width());
//            b = find_high(band, (1 - max_percent) * band.height() * band.width());
            auto a = find_low(band, min_percent * total_non_zero_pixel);
            auto b = find_high(band, (1 - max_percent) * total_non_zero_pixel);
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min, band);
            auto v1 = (band - c) / (d - c) * (max - min) + min;
            band = (band >= c & band <= d).ifthenelse(v1, band);
            result[i] = (band > d).ifthenelse(max, band);
            // ---------------------------------- sigmoid contrast ---------------------------------------------//
            auto temp = (result[i] > 0).ifthenelse(result[i], alpha);
            auto x = (beta * (alpha - temp)).exp();
            x = (x == 1).ifthenelse(0, x);
            result[i] = 1 / (1 + x);
            result[i] = (result[i] == 1).ifthenelse(0, result[i]);
        }
        if (channels == 3) {
//            auto t2 = x_get_current_ms();
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            vips_addalpha(dst.get_image(), *(&output), NULL);
        } else {
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            vips_copy(dst.get_image(), *(&output), NULL);
        }
    }


//    vips_addalpha(dst.get_image(), &output, NULL);
    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
    printf("done. sigmoid stretch...\n");
}

void
sigmoid_stretch_batch_calculate(char *input_name, char *output_name, double min, double max,
                                double min_adjust, double max_adjust, double alpha, double beta,
                                int order, const int *low, const int *high) {
    printf("entering sigmoid batch stretch...\n");
    auto start = x_get_current_ms();

    vips::VImage band, dst;

    double a, b, c, d;
    min = min / 255.;
    max = max / 255.;

    auto in = vips::VImage::new_from_file(input_name);
    auto channels = in.bands();
    vips::VImage result[channels];

    if (order == 1) {
        if (channels == 3) {
            in = in[2].bandjoin(in[1]).bandjoin(in[0]);
        } else if (channels == 4) {
            in = in[2].bandjoin(in[1]).bandjoin(in[0]).bandjoin(in[3]);
        }
    }

    auto input_zero = (in == 0).ifthenelse(0, 1);
    if (channels == 1) {
        a = low[0];
        b = high[0];
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        in = (in < c).ifthenelse(min, in);
        auto v1 = (in - c) / (d - c) * (max - min) + min;
        in = (in >= c & in <= d).ifthenelse(v1, in);
        dst = (in > d).ifthenelse(max, in);
        // ---------------------------------- sigmoid contrast ---------------------------------------------//
        auto temp = (dst > 0).ifthenelse(dst, alpha);
        auto x = (beta * (alpha - temp)).exp();
        x = (x == 1).ifthenelse(0, x);
        dst = 1 / (1 + x);
        dst = (dst == 1).ifthenelse(0, dst);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
    } else {
        for (int i = 0; i < channels; i++) {
            band = in.extract_band(i);
            a = low[i];
            b = high[i];
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min, band);
            auto v1 = (band - c) / (d - c) * (max - min) + min;
            band = (band >= c & band <= d).ifthenelse(v1, band);
            result[i] = (band > d).ifthenelse(max, band);
            // ---------------------------------- sigmoid contrast ---------------------------------------------//
            auto temp = (result[i] > 0).ifthenelse(result[i], alpha);
            auto x = (beta * (alpha - temp)).exp();
            x = (x == 1).ifthenelse(0, x);
            result[i] = 1 / (1 + x);
            result[i] = (result[i] == 1).ifthenelse(0, result[i]);
        }
    }

    switch (channels) {
        case 3:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        case 4:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        default:
            break;
    }

    save(dst, output_name);

    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
    printf("done. sigmoid batch stretch...\n");
}

void
sigmoid_stretch_batch_no_calculate(char *input_name, char *output_name, double min, double max,
                                   double min_percent, double max_percent, double min_adjust,
                                   double max_adjust, double alpha, double beta, int order) {
    printf("entering sigmoid batch NO calculate stretch...\n");
    auto start = x_get_current_ms();

    vips::VImage band, dst;

    double c, d;
    min = min / 255.;
    max = max / 255.;

    auto in = vips::VImage::new_from_file(input_name);
    auto channels = in.bands();
    vips::VImage result[channels];

    if (order == 1) {
        if (channels == 3) {
            in = in[2].bandjoin(in[1]).bandjoin(in[0]);
        } else if (channels == 4) {
            in = in[2].bandjoin(in[1]).bandjoin(in[0]).bandjoin(in[3]);
        }
    }

    auto input_zero = (in == 0).ifthenelse(0, 1);
    auto total_non_zero_pixel = x_count_non_zero_and_white(in);

    if (channels == 1) {
        auto a = find_low(in, min_percent * total_non_zero_pixel);
        auto b = find_high(in, (1 - max_percent) * total_non_zero_pixel);
        c = a - min_adjust * (b - a);
        d = b + max_adjust * (b - a);

        in = (in < c).ifthenelse(min, in);
        auto v1 = (in - c) / (d - c) * (max - min) + min;
        in = (in >= c & in <= d).ifthenelse(v1, in);
        dst = (in > d).ifthenelse(max, in);
        // ---------------------------------- sigmoid contrast ---------------------------------------------//
        auto temp = (dst > 0).ifthenelse(dst, alpha);
        auto x = (beta * (alpha - temp)).exp();
        x = (x == 1).ifthenelse(0, x);
        dst = 1 / (1 + x);
        dst = (dst == 1).ifthenelse(0, dst);
        dst = dst * input_zero;
        dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
    } else {
        for (int i = 0; i < channels; i++) {
            band = in.extract_band(i);
            auto a = find_low(band, min_percent * total_non_zero_pixel);
            auto b = find_high(band, (1 - max_percent) * total_non_zero_pixel);
            c = a - min_adjust * (b - a);
            d = b + max_adjust * (b - a);

            band = (band < c).ifthenelse(min, band);
            auto v1 = (band - c) / (d - c) * (max - min) + min;
            band = (band >= c & band <= d).ifthenelse(v1, band);
            result[i] = (band > d).ifthenelse(max, band);
            // ---------------------------------- sigmoid contrast ---------------------------------------------//
            auto temp = (result[i] > 0).ifthenelse(result[i], alpha);
            auto x = (beta * (alpha - temp)).exp();
            x = (x == 1).ifthenelse(0, x);
            result[i] = 1 / (1 + x);
            result[i] = (result[i] == 1).ifthenelse(0, result[i]);
        }
    }

    switch (channels) {
        case 3:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        case 4:
            dst = result[0].bandjoin(result[1]).bandjoin(result[2]).bandjoin(result[3]);
            dst = dst * input_zero;
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            break;
        default:
            break;
    }

    save(dst, output_name);

    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double) (end - start) / 1000);
    printf("done. sigmoid batch stretch NO calculate...\n");
}

VipsImage *memory_to_from(VipsImage *input) {
//    VipsImage *output;
    printf("entering memory to from...\n");
    auto start = x_get_current_ms();
    vips::VImage in, in1;
    void *buff;
    size_t size;

//    in = (vips::VImage) input;

    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&size);
    auto dst = vips::VImage::new_from_memory_steal(buff, size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);

//    buff = in.write_to_memory(&size);
//    auto dst = vips::VImage::new_from_memory(buff, size, in.width(), in.height(),
//                                             3, VIPS_FORMAT_DOUBLE);

//    in.write_to_buffer(".png", &buff, &size);
//    auto dst = vips::VImage::new_from_buffer(buff, size, "");

//    vips_addalpha(dst.get_image(), &output, NULL);

//    return output;
    auto end = x_get_current_ms();
    printf("memory to from computation time: %.8f second.\n", (double) (end - start) / 1000);
    printf("done. memory to from...\n");
    return dst.get_image();

}

VipsImage *hist_equalize(VipsImage *input) {
    VipsImage *output;
    vips::VImage in;
    cv::Mat src, dst, YCC;
    std::vector<cv::Mat> channels;

    in = vips::VImage(input, vips::NOSTEAL);

    src = x_vips_to_cv_8u(in);
    cv::cvtColor(src, YCC, cv::COLOR_BGR2YCrCb);
    cv::split(YCC, channels);
    cv::equalizeHist(channels[0], channels[0]);
    cv::merge(channels, YCC);
    cv::cvtColor(YCC, dst, cv::COLOR_YCrCb2BGR);

    auto out = x_cv_to_vips_double(dst);
    out = out.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(out.get_image(), &output, NULL);
    return output;
}

VipsImage *hist_equalize_clahe(VipsImage *input, double clip, int win) {
    VipsImage *output;
    vips::VImage in;
    cv::Mat src, dst, YCC;
    std::vector<cv::Mat> channels;

//    in = (vips::VImage) input;
    in = vips::VImage(input, vips::NOSTEAL);

    src = x_vips_to_cv_8u(in);
    cv::cvtColor(src, YCC, cv::COLOR_BGR2YCrCb);
    cv::split(YCC, channels);
    auto cl = cv::createCLAHE(clip, cv::Size(win, win));
    cl->apply(channels[0], channels[0]);

    cv::merge(channels, YCC);
    cv::cvtColor(YCC, dst, cv::COLOR_YCrCb2BGR);
    auto out = x_cv_to_vips_double(dst);
    out = out.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(out.get_image(), &output, NULL);
    return output;
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

VipsImage *haze_remove(VipsImage *input, int radius, double ratio) {
    VipsImage *output;
    vips::VImage in, in1, in2;
    size_t data_size;
    void *buff;
//    in = (vips::VImage) input;
    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
//    in = in2.cast(VIPS_FORMAT_DOUBLE);
//    in = (vips::VImage) input;
//    in = vips::VImage(input, vips::NOSTEAL);

    auto src = x_vips_to_cv_8u(in);
    auto darkChanelImg = dark_channel_image(src, radius);
    auto out = get_dehazed_img_guided_filter(src, darkChanelImg, ratio);
    auto dst = x_cv_to_vips_double(out);
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);

    return output;
}

VipsImage *double_filter(VipsImage *input, int radius, double sigmaColor, double sigmaSpace) {
    VipsImage *output;
    vips::VImage in;
    cv::Mat out;

    in = vips::VImage(input, vips::NOSTEAL);

    auto src = x_vips_to_cv_8u(in);
    cv::bilateralFilter(src, out, radius, sigmaColor, sigmaSpace);
    auto dst = x_cv_to_vips_double(out);
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);
    return output;
}

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

//    auto img_padding = x_cv_padding(input, padding_top, padding_bottom, padding_left, padding_right);
//    int pad_h = img_padding.rows;
//    int pad_w = img_padding.cols;
//    auto arma_padding = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) img_padding);
//    img_padding.release();
    auto img_padding = x_vips_padding(input, padding_top, padding_bottom, padding_left, padding_right);
    int pad_h = img_padding.height();
    int pad_w = img_padding.width();
    auto arma_padding = x_vips_to_arma_cube(img_padding);

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

//    auto cv_lmm = x_arma_to_cv(local_mean_map);
//    auto result = x_cv_to_vips_double(cv_lmm).mapim(idx);
//    cv_lmm.release();

    auto result = x_arma_cube_to_vips(local_mean_map).mapim(idx);
    local_mean_map.reset();

    return result;
}

vips::VImage polynomial_fitting(const vips::VImage &input, double p, double overlap) {
    vips::VImage out;

//    auto cv_input = x_vips_to_cv_64f(input);
//    auto arma_input = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) cv_input);
//    auto arma_stats = x_arma_cube_stats(arma_input);

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

//    auto img_padding = x_cv_padding(input, padding_top, padding_bottom, padding_left, padding_right);
//    auto arma_padding = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>) img_padding);
//    img_padding.release();

    auto img_padding = x_vips_padding(input, padding_top, padding_bottom, padding_left, padding_right);
    auto arma_padding = x_vips_to_arma_cube(img_padding);

    arma::cube local_mean_map(num_w, num_h, 3), lmm;
    arma::cube local_mean_center(num_w, num_h, 2), lmc;
    arma::cube tube;
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
            local_mean_center.tube(n, m) = (arma::vec) {center_w, center_h};
        }
    }


    lmm = local_mean_map.tube(1, 1, local_mean_map.n_rows - 1, local_mean_map.n_cols - 1);
    lmc = local_mean_center.tube(1, 1, local_mean_center.n_rows - 1, local_mean_center.n_cols - 1);
    lmc.slice(0) -= padding_top;
    lmc.slice(1) -= padding_left;

    arma::vec t_y = arma::linspace(0, h, h + 1);
    arma::mat t_y_mat = arma::reshape(t_y, 1, h);
    arma::mat t_idx_0 = arma::repmat(t_y_mat, w, 1);

    arma::vec t_x = arma::linspace(0, w, w + 1);
    arma::mat t_x_mat = arma::reshape(t_x, w, 1);
    arma::mat t_idx_1 = arma::repmat(t_x_mat, 1, h);

    arma::vec x1 = arma::vectorise(lmc.slice(0));
    arma::vec x2 = arma::vectorise(lmc.slice(1));

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
    arma::vec x8 = x1 % x4;
    arma::vec x9 = x2 % x3;
    x = arma::join_rows(x, x6);
    x = arma::join_rows(x, x7);
    x = arma::join_rows(x, x8);
    x = arma::join_rows(x, x9);

    arma::vec x_6 = arma::pow(x_1, 3);
    arma::vec x_7 = arma::pow(x_2, 3);
    arma::vec x_8 = x_1 % x_4;
    arma::vec x_9 = x_2 % x_3;

    xx = arma::join_rows(xx, x_6);
    xx = arma::join_rows(xx, x_7);
    xx = arma::join_rows(xx, x_8);
    xx = arma::join_rows(xx, x_9);

    arma::cube y = arma::reshape(lmm, lmm.n_rows * lmm.n_cols, 3, 1);
    arma::mat alpha = arma::inv(x.t() * x) * x.t() * y.slice(0);
    arma::mat target_surface = xx * alpha;

    arma::mat r = arma::reshape(target_surface.col(0), w, h);
    arma::mat g = arma::reshape(target_surface.col(1), w, h);
    arma::mat b = arma::reshape(target_surface.col(2), w, h);
    arma::cube dst = arma::join_slices(r, g);
    dst = arma::join_slices(dst, b);
//    auto stat = x_arma_cube_stats(dst);
//
//    cv::Mat cv_result = x_arma_to_cv(dst);
//    out = x_cv_to_vips_double(cv_result);
//    cv_result.release();
    out = x_arma_cube_to_vips(dst);

    return out;
}

VipsImage *color_transfer_3order(VipsImage *input, VipsImage *target, double p, double overlap, double alpha) {
    VipsImage *output;
    vips::VImage in, tar;
//    in = (vips::VImage) input;
//    tar = (vips::VImage) target;
    in = vips::VImage(input, vips::NOSTEAL);
    tar = vips::VImage(target, vips::NOSTEAL);

//    auto scale_opt = vips::VImage::option()->set("vscale", (double) in.height() / tar.height());
//    scale_opt->set("kernel", VIPS_KERNEL_LANCZOS3);
//    tar = tar.resize((double) in.width() / tar.width(), scale_opt);
    tar = tar.resize((double) in.width() / tar.width(), vips::VImage::option()->
            set("vscale", (double) in.height() / tar.height())->
            set("kernel", VIPS_KERNEL_LANCZOS3));

    auto local_mean_map = calculate_local_mean(in / 255, p, overlap);
    auto target_color_map = polynomial_fitting(tar / 255, p, overlap);

    auto gamma = target_color_map.log() / local_mean_map.log();
    auto dst = alpha * (in / 255).pow(gamma);
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);
    return output;
}

void color_transfer(VipsImage *input, VipsImage *target, VipsImage **output, double p, double overlap, double alpha) {
//    VipsImage *output;
    vips::VImage in, tar;
//    in = (vips::VImage) input;
//    tar = (vips::VImage) target;
    in = vips::VImage(input, vips::NOSTEAL);
    tar = vips::VImage(target, vips::NOSTEAL);


    tar = tar.resize((double) in.width() / tar.width(), vips::VImage::option()->
            set("vscale", (double) in.height() / tar.height())->
            set("kernel", VIPS_KERNEL_LANCZOS3));

    auto local_mean_map = calculate_local_mean(in / 255, p, overlap);
    auto target_color_map = calculate_local_mean(tar / 255, p, overlap);

    auto gamma = target_color_map.log() / local_mean_map.log();
    auto dst = alpha * (in / 255).pow(gamma);
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);


    vips_addalpha(dst.get_image(), &(*output), NULL);

//    return output;
}

int get_tfw_content(const char *filename, double *result) {
    std::ifstream ifs;
    std::string line;
    int i = 0;

    ifs.open(filename, std::ios::in);
    if (!ifs.is_open()) {
        return 1;
    }
    while (getline(ifs, line)) {
        if (!line.empty()) {
            result[i] = std::stod(line);
        }
        i += 1;
    }
    ifs.close();

    return 0;
}

vips::VImage get_3_bands_image(const char *filename) {
    auto img = vips::VImage::new_from_file(filename) / 255;
    if (img.bands() == 4) {
        return img[0].bandjoin(img[1]).bandjoin(img[2]);
    } else {
        return img;
    }
}

void global_adw_stride(const std::map<std::string, std::vector<long long>> &dict_tiff_to_base, const vips::VImage &base_img,
                       float p, float overlap, int full_h, int full_w, int img_num,
                       int *adw_size_base, int *adw_stride_base, int *adw_size_source, int *adw_stride_source) {
    vips::VImage source_img_data, base_img_data;
    int i = 0;

    using namespace arma;
    fmat images_stat_base(img_num, 3, fill::zeros);
    fmat images_stat_source(img_num, 3, fill::zeros);

    auto it = dict_tiff_to_base.begin();
    while (it != dict_tiff_to_base.end()) {
        source_img_data = vips::VImage::new_from_file(it->first.c_str()) / 255;
        base_img_data = base_img.extract_area(it->second[0], it->second[1], it->second[2], it->second[3]);
        base_img_data = base_img_data.resize((double) source_img_data.width() / it->second[2], vips::VImage::option()->
                set("vscale", (double) source_img_data.height() / it->second[3])->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255;

        images_stat_source(i, 0) = (float) source_img_data.avg();
        images_stat_base(i, 0) = (float) base_img_data.avg();

        auto temp_sum_source = (source_img_data.avg() - source_img_data).pow(2.0);
        auto temp_sum_base = (base_img_data.avg() - base_img_data).pow(2.0);
        auto img_sum_source = x_vips_to_arma_cube(temp_sum_source);
        auto img_sum_base = x_vips_to_arma_cube(temp_sum_base);
        fmat a_source = sum(conv_to<fcube>::from(img_sum_source), 2);
        fmat a_base = sum(conv_to<fcube>::from(img_sum_base), 2);
        images_stat_source(i, 1) = sum(sum(a_source)); // 组内方差
        images_stat_base(i, 1) = sum(sum(a_base)); // 组内方差

        images_stat_source(i, 2) = (float) source_img_data.width() * (float) source_img_data.height() * 3;
        images_stat_base(i, 2) = (float) base_img_data.width() * (float) base_img_data.height() * 3;
        i += 1;

        ++it;
    }

    auto images_mean_source = sum(images_stat_source.col(0) % images_stat_source.col(2)) / sum(images_stat_source.col(2));
    auto images_mean_base = sum(images_stat_base.col(0) % images_stat_base.col(2)) / sum(images_stat_base.col(2));

    auto images_SSA_source = sum(pow((images_stat_source.col(0) - images_mean_source), 2.0) % images_stat_source.col(2)); // 组间方差
    auto images_SSA_base = sum(pow((images_stat_base.col(0) - images_mean_base), 2.0) % images_stat_base.col(2)); // 组间方差

    auto images_SSE_source = sum(images_stat_source.col(1)); // 组内方差
    auto images_SSE_base = sum(images_stat_base.col(1)); // 组内方差

    auto images_SST_source = images_SSA_source + images_SSE_source; // 总方差
    auto images_SST_base = images_SSA_base + images_SSE_base; // 总方差

    auto images_std_source = sqrt(images_SST_source / sum(images_stat_source.col(2)));
    auto images_std_base = sqrt(images_SST_base / sum(images_stat_base.col(2)));

    float constant = 128.0 / 45;  //理想状态下的mean/std
    auto rho_source = p / images_std_source * images_mean_source / constant;  // std越大，ρ越小， ADWs的size小
    auto rho_base = p / images_std_base * images_mean_base / constant;  // std越大，ρ越小， ADWs的size小

    *adw_size_source = int(sqrt(rho_source * (float) full_h * rho_source * (float) full_w) / 2) * 2 + 1;  // 奇数
    *adw_size_base = int(sqrt(rho_base * (float) full_h * rho_base * (float) full_w) / 2) * 2 + 1;  // 奇数

    *adw_stride_source = int(float(*adw_size_source) * (1 - overlap) / 2) * 2;  // 偶数
    *adw_stride_base = int(float(*adw_size_base) * (1 - overlap) / 2) * 2;  // 偶数

}

void get_query_tables(const char *base_tfw_name, const char *source_dir,
                      std::map<std::string, std::string> *dict_tiffs_tfws,
                      std::map<std::string, std::string> *query_table_geo_tif,
                      std::map<std::string, std::vector<long long>> *dict_tiff_to_base,
                      std::map<std::string, std::vector<long long>> *query_table_tif_geo,
                      int *full_h, int *full_w, int *img_num) {

    double source_tfw_content[6], base_tfw_content[6];
    int scale = 10000;
    int full_height = 0, full_width = 0, images_num = 0;

    if (get_tfw_content(base_tfw_name, &base_tfw_content[0]) != 0) {
        printf("error reading base tfw file...\n");
        return;
    }

    for (const auto &entry: std::filesystem::directory_iterator(std::filesystem::u8path(source_dir))) {
        auto file_name = entry.path().string();
        auto ext = entry.path().extension();

        if (ext == ".tif" || ext == ".tiff" || ext == ".TIF" || ext == ".TIFF") {
            auto tif_img_data = vips::VImage::new_from_file(file_name.c_str());
            auto tfw_name = entry.path().parent_path().string() + "\\" + entry.path().stem().string() + ".tfw";
            /*
             * source图的tiff到配套tfw的映射关系，key和value都保存的是带目录的文件名[dict_tiffs_tfws]
             * string(tif):string(tfw)
             */
            dict_tiffs_tfws->insert(std::map<std::string, std::string>::value_type(file_name, tfw_name));
            if (get_tfw_content(tfw_name.c_str(), &source_tfw_content[0]) != 0) {
                printf("error reading source tfw file...\n");
                return;
            }
            /*
             * tiff 影像到小地图上的映射表构建[dict_tiff_to_base] string:[x, y, w, h]
             * Key：带目录的tif文件名
             * Value：[x, y, w, h], 要裁切的左上角坐标x, y, 要在地图上裁切的宽高w, h
             */
            auto zeroX = base_tfw_content[4] - base_tfw_content[0] / 2;
            auto zeroY = base_tfw_content[5] + base_tfw_content[0] / 2;
            auto leftX = (long long) ((source_tfw_content[4] - source_tfw_content[0] / 2 - zeroX) / base_tfw_content[0]);
            auto leftY = (long long) ((zeroY - (source_tfw_content[5] + source_tfw_content[0] / 2)) / base_tfw_content[0]);
            auto rightX = (long long) ((source_tfw_content[4] - source_tfw_content[0] / 2 + source_tfw_content[0] * tif_img_data.width() - zeroX) / base_tfw_content[0]);
            auto rightY = (long long) ((zeroY - (source_tfw_content[5] + source_tfw_content[0] / 2 - source_tfw_content[0] * tif_img_data.height())) / base_tfw_content[0]);
            std::vector<long long> v = {leftX, leftY, rightX - leftX, rightY - leftY};
            dict_tiff_to_base->insert(std::map<std::string, std::vector<long long>>::value_type(file_name, v));
            /*
             * 查询表1：query_table_geo_tif, string:string
             * Key: 该tif文件的坐标，用下划线拼接，如：{“500086_4071725”: xxx.tif}
             * Value: 带目录的tif文件名
             *
             * 查询表2：query_table_tif_geo, string:[geo_w, geo_h]
             * Key: 带目录的tif文件名
             * Value: 一个2元素的vector数组geo_w, geo_h : { xxx.tif：[500086, 4071725]}
             */
            auto x0 = (long long) ((source_tfw_content[4] - source_tfw_content[0] / 2) * scale);
            auto y0 = (long long) ((source_tfw_content[5] + source_tfw_content[0] / 2) * scale);
            auto xn = (long long) ((source_tfw_content[4] - source_tfw_content[0] / 2 + source_tfw_content[0] * tif_img_data.width()) * scale);
            auto yn = (long long) ((source_tfw_content[5] + source_tfw_content[0] / 2 - source_tfw_content[0] * tif_img_data.height()) * scale);

            auto geo_h = y0 - yn;
            auto geo_w = xn - x0;
            auto x0_y0 = std::to_string(x0) + "_" + std::to_string(y0);
//            printf("geo_tif [%s, %s]\n", x0_y0.c_str(), file_name.c_str());
            query_table_geo_tif->insert(std::map<std::string, std::string>::value_type(x0_y0, file_name));
            std::vector<long long> v2 = {x0, y0, geo_w, geo_h};
            query_table_tif_geo->insert(std::map<std::string, std::vector<long long>>::value_type(file_name, v2));

            // 累加返回值
            images_num += 1;
            full_height += tif_img_data.height();
            full_width += tif_img_data.width();
        }
    }
    *full_h = full_height;
    *full_w = full_width;
    *img_num = images_num;
}

void padding_calculation(int h, int w, int adw_size, int adw_stride, int *top, int *bottom, int *left, int *right, int *number_h, int *number_w) {
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
    if (padding_top <= int(adw_size / 2)) {
        padding_top += int(adw_stride / 2);
        padding_bottom += int(adw_stride / 2);
        num_h += 1;
    }
    if (padding_left <= int(adw_size / 2)) {
        padding_left += int(adw_stride / 2);
        padding_right += int(adw_stride / 2);
        num_w += 1;
    }

    *top = padding_top;
    *bottom = padding_bottom;
    *left = padding_left;
    *right = padding_right;
    *number_h = num_h;
    *number_w = num_w;
}

vips::VImage local_mean_map_calculation(const vips::VImage &image_padding, const int num_h, const int num_w, const int adw_size,
                                        const int adw_stride, const int img_h, const int img_w) {
    auto arma_padding = x_vips_to_arma_cube(image_padding);
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
    int pad_h = image_padding.height();
    int pad_w = image_padding.width();
    auto m_h_ = pad_h - (adw_size - 1);
    auto m_w_ = pad_w - (adw_size - 1);

    auto top_x = int((m_w_ - img_w) / 2);
    auto top_y = int((m_h_ - img_h) / 2);
    auto idx = vips::VImage::xyz(m_w_, m_h_);
    idx = idx / adw_stride;
    idx = idx.extract_area(top_x, top_y, img_w, img_h);

//    printf("image w, h [%d, %d, %d, %d]\n", top_x, top_y, img_w, img_h);

    auto result = x_arma_cube_to_vips(local_mean_map).mapim(idx);

    return result;
}

std::vector<vips::VImage> color_transfer_padding(const vips::VImage &img_data_source,
                                                 const vips::VImage &img_data_base,
                                                 const char *img_file_name_source,
                                                 const std::map<std::string, std::vector<long long>> &query_table_tif_geo,
                                                 const std::map<std::string, std::vector<long long>> &dict_tiff_to_base,
                                                 const std::map<std::string, std::string> &query_table_geo_tif,
                                                 int adw_size_source, int adw_stride_source, int adw_size_base, int adw_stride_base) {
    int num_h_source, num_w_source, num_h_base, num_w_base;
    // 准备base影像裁切和resize
    auto c = dict_tiff_to_base.find(img_file_name_source);
    auto cut_base_img = img_data_base.extract_area(c->second[0], c->second[1], c->second[2], c->second[3]);
    cut_base_img = cut_base_img.resize((double) img_data_source.width() / cut_base_img.width(), vips::VImage::option()->
            set("vscale", (double) img_data_source.height() / cut_base_img.height())->
            set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
//    printf("resize wh [%d, %d]\n", cut_base_img.width(), cut_base_img.height());

    int padding_top_source, padding_bottom_source, padding_left_source, padding_right_source;
    int padding_top_base, padding_bottom_base, padding_left_base, padding_right_base;

    padding_calculation(img_data_source.height(), img_data_source.width(), adw_size_source, adw_stride_source, &padding_top_source,
                        &padding_bottom_source, &padding_left_source, &padding_right_source, &num_h_source, &num_w_source);
    padding_calculation(cut_base_img.height(), cut_base_img.width(), adw_size_base, adw_stride_base, &padding_top_base, &padding_bottom_base,
                        &padding_left_base, &padding_right_base, &num_h_base, &num_w_base);

    auto a = query_table_tif_geo.find(img_file_name_source);
    auto x0 = a->second[0];
    auto y0 = a->second[1];
    auto geo_w = a->second[2];
    auto geo_h = a->second[3];

    auto key_top_left = std::to_string(x0 - geo_w) + "_" + std::to_string(y0 + geo_h);
    auto key_top = std::to_string(x0) + "_" + std::to_string(y0 + geo_h);
    auto key_top_right = std::to_string(x0 + geo_w) + "_" + std::to_string(y0 + geo_h);
    auto key_left = std::to_string(x0 - geo_w) + "_" + std::to_string(y0);
    auto key_right = std::to_string(x0 + geo_w) + "_" + std::to_string(y0);
    auto key_bottom_left = std::to_string(x0 - geo_w) + "_" + std::to_string(y0 - geo_h);
    auto key_bottom = std::to_string(x0) + "_" + std::to_string(y0 - geo_h);
    auto key_bottom_right = std::to_string(x0 + geo_w) + "_" + std::to_string(y0 - geo_h);


    vips::VImage img_top_left_source, img_top_source, img_top_right_source, img_left_source,
            img_right_source, img_bottom_left_source, img_bottom_source, img_bottom_right_source;

    vips::VImage img_top_left_base, img_top_base, img_top_right_base, img_left_base,
            img_right_base, img_bottom_left_base, img_bottom_base, img_bottom_right_base;

    vips::VImage image_padding_source = vips::VImage::black(img_data_source.width() + padding_left_source + padding_right_source,
                                                            img_data_source.height() + padding_top_source + padding_bottom_source);

    vips::VImage image_padding_base = vips::VImage::black(cut_base_img.width() + padding_left_base + padding_right_base,
                                                          cut_base_img.height() + padding_top_base + padding_bottom_base);


    auto flip_horizontal_source = img_data_source.flip(VIPS_DIRECTION_HORIZONTAL);
    auto flip_vertical_source = img_data_source.flip(VIPS_DIRECTION_VERTICAL);
    auto flip_twice_source = img_data_source.flip(VIPS_DIRECTION_HORIZONTAL);
    flip_twice_source = flip_twice_source.flip(VIPS_DIRECTION_VERTICAL);

    auto flip_horizontal_base = cut_base_img.flip(VIPS_DIRECTION_HORIZONTAL);
    auto flip_vertical_base = cut_base_img.flip(VIPS_DIRECTION_VERTICAL);
    auto flip_twice_base = cut_base_img.flip(VIPS_DIRECTION_HORIZONTAL);
    flip_twice_base = flip_twice_base.flip(VIPS_DIRECTION_VERTICAL);

    // ---------------------------上面一排-------------------------------- //
    auto end = query_table_geo_tif.end();
    // 左上角
    auto b = query_table_geo_tif.find(key_top_left);
    if (b != end) {
        img_top_left_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_top_left_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_top_left_base = img_top_left_base.resize((double) cut_base_img.width() / img_top_left_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_top_left_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_top_left_source = flip_twice_source;
        img_top_left_base = flip_twice_base;
    }
    image_padding_source = image_padding_source.insert(img_top_left_source.extract_area(img_top_left_source.width() - padding_left_source,
                                                                                        img_top_left_source.height() - padding_top_source,
                                                                                        padding_left_source, padding_top_source),
                                                       0, 0);
    image_padding_base = image_padding_base.insert(img_top_left_base.extract_area(img_top_left_base.width() - padding_left_base,
                                                                                  img_top_left_base.height() - padding_top_base,
                                                                                  padding_left_base, padding_top_base),
                                                   0, 0);

    // 上面
    b = query_table_geo_tif.find(key_top);
    if (b != end) {
        img_top_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_top_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_top_base = img_top_base.resize((double) cut_base_img.width() / img_top_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_top_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_top_source = flip_vertical_source;
        img_top_base = flip_vertical_base;
    }
    image_padding_source = image_padding_source.insert(img_top_source.extract_area(0, img_top_source.height() - padding_top_source,
                                                                                   img_top_source.width(), padding_top_source),
                                                       padding_left_source, 0);
    image_padding_base = image_padding_base.insert(img_top_base.extract_area(0, img_top_base.height() - padding_top_base,
                                                                             img_top_base.width(), padding_top_base),
                                                   padding_left_base, 0);

    // 右上角
    b = query_table_geo_tif.find(key_top_right);
    if (b != end) {
        img_top_right_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_top_right_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_top_right_base = img_top_right_base.resize((double) cut_base_img.width() / img_top_right_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_top_right_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_top_right_source = flip_twice_source;
        img_top_right_base = flip_twice_base;
    }
    image_padding_source = image_padding_source.insert(img_top_right_source.extract_area(0, img_top_right_source.height() - padding_top_source,
                                                                                         padding_right_source, padding_top_source),
                                                       padding_left_source + img_data_source.width(), 0);
    image_padding_base = image_padding_base.insert(img_top_right_base.extract_area(0, img_top_right_base.height() - padding_top_base,
                                                                                   padding_right_base, padding_top_base),
                                                   padding_left_base + cut_base_img.width(), 0);

    // ---------------------------中间一排-------------------------------- //
    // 左边
    b = query_table_geo_tif.find(key_left);
    if (b != end) {
        img_left_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_left_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_left_base = img_left_base.resize((double) cut_base_img.width() / img_left_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_left_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_left_source = flip_horizontal_source;
        img_left_base = flip_horizontal_base;
    }
    image_padding_source = image_padding_source.insert(img_left_source.extract_area(img_left_source.width() - padding_left_source,
                                                                                    0, padding_left_source, img_left_source.height()),
                                                       0, padding_top_source);
    image_padding_base = image_padding_base.insert(img_left_base.extract_area(img_left_base.width() - padding_left_base,
                                                                              0, padding_left_base, img_left_base.height()),
                                                   0, padding_top_base);


//    // 加入自己，居中那张
    image_padding_source = image_padding_source.insert(img_data_source, padding_left_source, padding_top_source);
    image_padding_base = image_padding_base.insert(cut_base_img, padding_left_base, padding_top_base);

    // 右边
    b = query_table_geo_tif.find(key_right);
    if (b != end) {
        img_right_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_right_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_right_base = img_right_base.resize((double) cut_base_img.width() / img_right_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_right_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_right_source = flip_horizontal_source;
        img_right_base = flip_horizontal_base;
    }
    image_padding_source = image_padding_source.insert(img_right_source.extract_area(0, 0, padding_right_source, img_right_source.height()),
                                                       padding_left_source + img_data_source.width(), padding_top_source);
    image_padding_base = image_padding_base.insert(img_right_base.extract_area(0, 0, padding_right_base, img_right_base.height()),
                                                   padding_left_base + cut_base_img.width(), padding_top_base);


    // ---------------------------下面一排-------------------------------- //

//     左下角
    b = query_table_geo_tif.find(key_bottom_left);
    if (b != end) {
        img_bottom_left_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_bottom_left_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_bottom_left_base = img_bottom_left_base.resize((double) cut_base_img.width() / img_bottom_left_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_bottom_left_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_bottom_left_source = flip_twice_source;
        img_bottom_left_base = flip_twice_base;
    }
    image_padding_source = image_padding_source.insert(img_bottom_left_source.extract_area(img_bottom_left_source.width() - padding_left_source,
                                                                                           0, padding_left_source, padding_bottom_source),
                                                       0, img_data_source.height() + padding_bottom_source);

    image_padding_base = image_padding_base.insert(img_bottom_left_base.extract_area(img_bottom_left_base.width() - padding_left_base,
                                                                                     0, padding_left_base, padding_bottom_base),
                                                   0, cut_base_img.height() + padding_bottom_base);


//    // 下面
    b = query_table_geo_tif.find(key_bottom);
    if (b != end) {
        img_bottom_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_bottom_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_bottom_base = img_bottom_base.resize((double) cut_base_img.width() / img_bottom_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_bottom_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_bottom_source = flip_vertical_source;
        img_bottom_base = flip_vertical_base;
    }
    image_padding_source = image_padding_source.insert(img_bottom_source.extract_area(0, 0, img_bottom_source.width(), padding_bottom_source),
                                                       padding_left_source, img_data_source.height() + padding_top_source);

    image_padding_base = image_padding_base.insert(img_bottom_base.extract_area(0, 0, img_bottom_base.width(), padding_bottom_base),
                                                   padding_left_base, cut_base_img.height() + padding_top_base);


//    // 右下角
    b = query_table_geo_tif.find(key_bottom_right);
    if (b != end) {
        img_bottom_right_source = get_3_bands_image(b->second.c_str());
        auto d = dict_tiff_to_base.find(b->second);
        img_bottom_right_base = img_data_base.extract_area(d->second[0], d->second[1], d->second[2], d->second[3]);
        img_bottom_right_base = img_bottom_right_base.resize((double) cut_base_img.width() / img_bottom_right_base.width(), vips::VImage::option()->
                set("vscale", (double) cut_base_img.height() / img_bottom_right_base.height())->
                set("kernel", VIPS_KERNEL_LINEAR)) / 255.;
    } else {
        img_bottom_right_source = flip_twice_source;
        img_bottom_right_base = flip_twice_base;
    }
    image_padding_source = image_padding_source.insert(img_bottom_right_source.extract_area(0, 0, padding_right_source, padding_bottom_source),
                                                       img_data_source.width() + padding_left_source, img_data_source.height() + padding_top_source);

    image_padding_base = image_padding_base.insert(img_bottom_right_base.extract_area(0, 0, padding_right_base, padding_bottom_base),
                                                   cut_base_img.width() + padding_left_base, cut_base_img.height() + padding_top_base);


    // 开始计算local mean map
    auto local_mean_source = local_mean_map_calculation(image_padding_source, num_h_source, num_w_source, adw_size_source, adw_stride_source,
                                                        img_data_source.height(), img_data_source.width());
    auto local_mean_base = local_mean_map_calculation(image_padding_base, num_h_base, num_w_base, adw_size_base, adw_stride_base,
                                                      cut_base_img.height(), cut_base_img.width());


    return {local_mean_source, local_mean_base};

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

void color_transfer_no_joint_batch(const char *base_name, const char *input_dir, const char *output_dir,
                                   float p, float overlap, float alpha) {

    auto base_image = vips::VImage::new_from_file(base_name);
    std::map<std::string, std::string> dict_tiffs_tfws, query_table_geo_tif;
    std::map<std::string, std::vector<long long>> dict_tiff_to_base, query_table_tif_geo;

    std::filesystem::path base_path(base_name);
    auto base_tfw_name = base_path.parent_path().string() + "\\" + base_path.stem().string() + ".tfw";

    // 1. get query tables
    int full_h, full_w, img_num;
    get_query_tables(base_tfw_name.c_str(), input_dir, &dict_tiffs_tfws, &query_table_geo_tif, &dict_tiff_to_base,
                     &query_table_tif_geo, &full_h, &full_w, &img_num);

    // 2. compute adw size, stride
    int adw_size_base, adw_stride_base, adw_size_source, adw_stride_source;
    auto start = x_get_current_ms();
    global_adw_stride(dict_tiff_to_base, base_image, p, overlap, full_h, full_w, img_num,
                      &adw_size_base, &adw_stride_base, &adw_size_source, &adw_stride_source);
    auto end = x_get_current_ms();
//    printf("adw calculation time : %f second\n", double(end - start) / 1000);

//    // 3. color transfer
    for (const auto &entry: std::filesystem::directory_iterator(input_dir)) {
        vips::VImage band4;

        auto ext = entry.path().extension();
        if (ext == ".tif" || ext == ".tiff" || ext == ".TIF" || ext == ".TIFF") {
            auto img_data = vips::VImage::new_from_file(entry.path().string().c_str()) / 255.;
            auto bands = img_data.bands();
            if (bands == 4) {
                band4 = img_data[3];
            }
            img_data = img_data[0].bandjoin(img_data[1]).bandjoin(img_data[2]);
            auto local_mean_maps = color_transfer_padding(img_data, base_image, entry.path().string().c_str(), query_table_tif_geo, dict_tiff_to_base,
                                                          query_table_geo_tif, adw_size_source, adw_stride_source, adw_size_base, adw_stride_base);

            auto gamma = local_mean_maps[1].log() / local_mean_maps[0].log();
            auto dst = alpha * img_data.pow(gamma);
            if (bands == 4) {
                dst = dst.bandjoin(band4);
            }
            dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
            auto output_name = std::string(output_dir).append("\\").append(entry.path().filename().string());
            save(dst, output_name.c_str());

            // entry.path().string(): D:\xmap_test_imagedata\mapping\seqian\ys\J50H004084.tif
            auto it = dict_tiffs_tfws.find(entry.path().string());
            auto tfw_name = it->second.c_str();
            std::filesystem::path tfw_path(tfw_name);
            auto output_tfw_name = std::string(output_dir) + "\\" + tfw_path.stem().string() + ".tfw";
//            printf("saving tfw file... %s\n", output_tfw_name.c_str());
            copy_tfw_file(tfw_name, output_tfw_name.c_str());
        }
    }
}


double vips_image_min(VipsImage *input) {
    double out;
    vips_min(input, &out, NULL);

    return out;
}

double vips_image_max(VipsImage *input) {
    double out;

    vips_max(input, &out, NULL);

    return out;
}

void save_tiff(VipsImage *input, const char *filename, int compression, int quality) {
    printf("entering save tif...\n");
    auto start = x_get_current_ms();
    VipsForeignTiffCompression level;
    vips::VImage in, in1, in2;
    size_t data_size;
    void *buff;

//    in1 = (vips::VImage) input;
    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in = vips::VImage::new_from_memory(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);

    switch (compression) {
        case 0:
            level = VIPS_FOREIGN_TIFF_COMPRESSION_NONE;
            break;
        case 1:
            level = VIPS_FOREIGN_TIFF_COMPRESSION_LZW;
            break;
        case 2:
            level = VIPS_FOREIGN_TIFF_COMPRESSION_JPEG;
            break;
        default:
            level = VIPS_FOREIGN_TIFF_COMPRESSION_NONE;
    }
//    auto opt = vips::VImage::option()->set("compression", level);
//    opt->set("Q", quality);
//    in.tiffsave(filename, opt);

    in.write_to_file(filename, vips::VImage::option()->
            set("compression", level)->
            set("Q", quality));

    auto end = x_get_current_ms();
    printf("save tiff time: %.8f second.\n", (double) (end - start) / 1000);
    printf("done. save tiff...\n");
}

void calculate_hist_total(const char *directory, int bands, int order, double min_p, double max_p, int *a, int *b) {
    vips::VImage img;
//    int pixel_number = 0;
    double pixel_number = 0.0;
    vips::VImage hist_total[bands];
    double min_total, max_total, T1, T2;
    int i, j;


    img = vips::VImage::black(10, 10);
    for (int k = 0; k < bands; k++) {
        hist_total[k] = img.hist_find() * 0;
    }

    // 计算所有的直方图累加
    for (const auto &entry: std::filesystem::directory_iterator(directory)) {
        auto filename = entry.path().string();
        size_t pos = filename.find('.', 0);
        auto suffix = filename.substr(pos + 1, 3);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
        if (suffix == "tif") {
            img = vips::VImage::new_from_file(filename.c_str());
            if (order == 1) {
                if (bands == 3) {
                    img = img[2].bandjoin(img[1]).bandjoin(img[0]);
                } else if (bands == 4) {
                    img = img[2].bandjoin(img[1]).bandjoin(img[0]).bandjoin(img[3]);
                }
            }

            for (int k = 0; k < bands; k++) {
                if (bands == 1) {
                    hist_total[k] += img.hist_find();
                } else {
                    hist_total[k] += img[k].hist_find();
                }
            }
            /* 2022-07-22：反馈bug，这里统计像素的时候应该排除掉0像素
             * 添加一个函数count_non_zero，统计3波段影像非零像素个数
             * 未来有可能还会去掉65535像素值
            */
//            pixel_number += img.width() * img.height();
            /*
             * 2022-07-29：在济南讨论，决定把所有非零和非65535的像素都排除掉
             */
            pixel_number += x_count_non_zero_and_white(img);
        } else if (suffix == "jpg") {
            img = vips::VImage::new_from_file(filename.c_str());
            if (order == 1) {
                if (bands == 3) {
                    img = img[2].bandjoin(img[1]).bandjoin(img[0]);
                } else if (bands == 4) {
                    img = img[2].bandjoin(img[1]).bandjoin(img[0]).bandjoin(img[3]);
                }
            }

            for (int k = 0; k < bands; k++) {
                if (bands == 1) {
                    hist_total[k] += img.hist_find();
                } else {
                    hist_total[k] += img[k].hist_find();
                }
            }

            pixel_number += x_count_non_zero_and_white(img);
        }
    }

    // 找到边界的a、b值
    T1 = min_p * pixel_number;
    T2 = (1 - max_p) * pixel_number;

    for (int m = 0; m < bands; m++) {
        min_total = 0.0;
        max_total = 0.0;
        i = 1; // i从1开始，是因为要排除掉0值；也就是说0的像素多少跟后面计算无关

        if (bands == 1) {
            // --- find a:
            while (min_total <= T1) {
                min_total += hist_total[m](i, 0)[0];
                i += 1;
            }
            a[m] = i - 1; // a值
            // --- find b:
            auto w = hist_total[m].width();
            if (w == 65536) {
                j = w - 2;
            } else { j = w - 1; }   // 如果hist的宽度是65536，说明有白色要去掉，w-1就是65535的个数，w-2就是到65534，相当于白色剔除掉；

            while (max_total <= T2) {
                max_total += hist_total[m](j, 0)[0];
                j -= 1;
            }
            b[m] = j + 1; // b值
        } else {
            // --- find a:
            while (min_total <= T1) {
                min_total += hist_total[m](i, 0)[0];
                i += 1;
            }
            a[m] = i - 1; // a值
            // --- find b:
            auto w = hist_total[m].width();
            if (w == 65536) {
                j = w - 2;
            } else { j = w - 1; }   // 如果hist的宽度是65536，说明有白色要去掉，w-1就是65535的个数，w-2就是到65534，相当于白色剔除掉；

            while (max_total <= T2) {
                max_total += hist_total[m](j, 0)[0];
                j -= 1;
            }
            b[m] = j + 1; // b值
        }
    }
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

void AGCIE(const cv::Mat &src, cv::Mat &dst) {
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

    cv::Mat L_norm;
    L.convertTo(L_norm, CV_64F, 1.0 / 255.0);

    cv::Mat mean, stddev;
    cv::meanStdDev(L_norm, mean, stddev);
    double mu = mean.at<double>(0, 0);
    double sigma = stddev.at<double>(0, 0);

    double tau = 3.0;

    double gamma;
    if (4 * sigma <= 1.0 / tau) { // low-contrast
        gamma = -std::log2(sigma);
    } else { // high-contrast
        gamma = std::exp((1.0 - mu - sigma) / 2.0);
    }

    std::vector<double> table_double(256, 0);
    for (int i = 1; i < 256; i++) {
        table_double[i] = i / 255.0;
    }

    if (mu >= 0.5) { // bright image
        for (int i = 1; i < 256; i++) {
            table_double[i] = std::pow(table_double[i], gamma);
        }
    } else { // dark image
        double mu_gamma = std::pow(mu, gamma);
        for (int i = 1; i < 256; i++) {
            double in_gamma = std::pow(table_double[i], gamma);;
            table_double[i] = in_gamma / (in_gamma + (1.0 - in_gamma) * mu_gamma);
        }
    }

    std::vector<uchar> table_uchar(256, 0);
    for (int i = 1; i < 256; i++) {
        table_uchar[i] = cv::saturate_cast<uchar>(255.0 * table_double[i]);
    }

    cv::LUT(L, table_uchar, L);

    if (channels == 1) {
        dst = L.clone();
    } else {
        cv::merge(HSV_channels, dst);
        cv::cvtColor(dst, dst, cv::COLOR_HSV2BGR_FULL);
    }

}

void WTHE(const cv::Mat &src, cv::Mat &dst, double r, double v) {
    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();
    int total_pixels = rows * cols;

    cv::Mat L;
    cv::Mat YUV;
    std::vector<cv::Mat> YUV_channels;
    if (channels == 1) {
        L = src.clone();
    } else {
        cv::cvtColor(src, YUV, cv::COLOR_BGR2YUV);
        cv::split(YUV, YUV_channels);
        L = YUV_channels[0];
    }

    int histsize = 256;
    float range[] = {0, 256};
    const float *histRanges = {range};
    int bins = 256;
    cv::Mat hist;
    calcHist(&L, 1, 0, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);

    float total_pixels_inv = 1.0f / total_pixels;
    cv::Mat P = hist.clone();
    for (int i = 0; i < 256; i++) {
        P.at<float>(i) = P.at<float>(i) * total_pixels_inv;
    }

    cv::Mat Pwt = P.clone();
    double minP, maxP;
    cv::minMaxLoc(P, &minP, &maxP);
    float Pu = v * maxP;
    float Pl = minP;
    for (int i = 0; i < 256; i++) {
        float Pi = P.at<float>(i);
        if (Pi > Pu)
            Pwt.at<float>(i) = Pu;
        else if (Pi < Pl)
            Pwt.at<float>(i) = 0;
        else
            Pwt.at<float>(i) = std::pow((Pi - Pl) / (Pu - Pl), r) * Pu;
    }

    cv::Mat Cwt = Pwt.clone();
    float cdf = 0;
    for (int i = 0; i < 256; i++) {
        cdf += Pwt.at<float>(i);
        Cwt.at<float>(i) = cdf;
    }

    float Wout = 255.0f;
    float Madj = 0.0f;
    std::vector<uchar> table(256, 0);
    for (int i = 0; i < 256; i++) {
        table[i] = cv::saturate_cast<uchar>(Wout * Cwt.at<float>(i) + Madj);
    }

    cv::LUT(L, table, L);

    if (channels == 1) {
        dst = L.clone();
    } else {
        cv::merge(YUV_channels, dst);
        cv::cvtColor(dst, dst, cv::COLOR_YUV2BGR);
    }

}

VipsImage *vips_wthe(VipsImage *input, double r, double v) {
    VipsImage *output;
    vips::VImage in, in1, in2;
    size_t data_size;
    void *buff;
//    in = (vips::VImage) input;
    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
//    in = in2.cast(VIPS_FORMAT_DOUBLE);
//    in = (vips::VImage) input;
//    in = vips::VImage(input, vips::NOSTEAL);

    auto src = x_vips_to_cv_8u(in);
    cv::Mat out;
    WTHE(src, out, r, v);
    auto dst = x_cv_to_vips_double(out);
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);

    return output;
}

VipsImage *vips_agcie(VipsImage *input) {
    VipsImage *output;
    vips::VImage in, in1, in2;
    size_t data_size;
    void *buff;
//    in = (vips::VImage) input;
    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
//    in = in2.cast(VIPS_FORMAT_DOUBLE);
//    in = (vips::VImage) input;
//    in = vips::VImage(input, vips::NOSTEAL);

    auto src = x_vips_to_cv_8u(in);
    cv::Mat out;
    AGCIE(src, out);
    auto dst = x_cv_to_vips_double(out);
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);

    return output;
}

VipsImage *vips_agcwd(VipsImage *input, double alpha) {
    VipsImage *output;
    vips::VImage in, in1, in2;
    size_t data_size;
    void *buff;
//    in = (vips::VImage) input;
    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
//    in = in2.cast(VIPS_FORMAT_DOUBLE);
//    in = (vips::VImage) input;
//    in = vips::VImage(input, vips::NOSTEAL);

    auto src = x_vips_to_cv_8u(in);
    cv::Mat out;
    AGCWD(src, out, alpha);
    auto dst = x_cv_to_vips_double(out);
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);

    return output;
}

void contrast_agcwd_batch(const char *input_name, const char *output_name, float alpha, int keep_nir) {
    vips::VImage output, in;

    auto input = vips::VImage::new_from_file(input_name);
    auto input_zero = (input == 0).ifthenelse(0, 1);

    auto src = x_vips_to_cv_8u(input);
    cv::Mat out;
    AGCWD(src, out, alpha);
    auto dst = x_cv_to_vips_double(out);
//    dst = dst.cast(VIPS_FORMAT_UCHAR);

    dst = dst / 255.;
    if (keep_nir == 4) {
        dst = dst.bandjoin(input[3] / 255.);
    }

//    printf("dst bands=%d\n", dst.bands());
    if (dst.bands() == 4) {
        dst = dst * input_zero;
    } else {
        dst = dst * (input_zero[0].bandjoin(input_zero[1]).bandjoin(input_zero[2]));
    }
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);
    save(dst, output_name);
}

VipsImage *vips_aindane(VipsImage *input, int sigma1, int sigma2, int sigma3) {
    VipsImage *output;
    vips::VImage in, in1, in2;
    size_t data_size;
    void *buff;
//    in = (vips::VImage) input;
    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);
//    in = in2.cast(VIPS_FORMAT_DOUBLE);
//    in = (vips::VImage) input;
//    in = vips::VImage(input, vips::NOSTEAL);

    auto src = x_vips_to_cv_8u(in);
    cv::Mat out;
    AINDANE(src, out, sigma1, sigma2, sigma3);
    auto dst = x_cv_to_vips_double(out);
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    vips_addalpha(dst.get_image(), &output, NULL);

    return output;
}

void enhance_green_band(VipsImage *input, VipsImage **output, float ndvi_ratio, float green_ratio) {
    vips::VImage in, in1, in2, dst;
    size_t data_size;
    void *buff;

    in1 = vips::VImage(input, vips::NOSTEAL);
    auto h = in1.height();
    auto w = in1.width();
    buff = in1.write_to_memory(&data_size);
    in = vips::VImage::new_from_memory_steal(buff, data_size, w, h, in1.bands(), VIPS_FORMAT_UCHAR);

    in = in / 255.;
    auto input_zero = (in == 0).ifthenelse(0, 1);

    auto ndvi = (in[3] - in[0]) / (in[3] + in[0]);
    auto greater = (ndvi > ndvi_ratio).ifthenelse(1, 0);
    auto less = (ndvi <= ndvi_ratio).ifthenelse(1, 0);
    auto green_new = greater * (in[1] * green_ratio + in[3] * (1 - green_ratio)) + less * in[1];

    dst = in[0].bandjoin(green_new).bandjoin(in[2]).bandjoin(in[3]);
    dst = dst * input_zero;
    dst = (dst * 255).cast(VIPS_FORMAT_UCHAR);

    vips_copy(dst.get_image(), *(&output), NULL);
}

int enhance_green_band_batch(const char *input_name, const char *output_name, float ndvi_ratio,
                             float green_ratio, int keep_nir) {
    vips::VImage output;

    auto input = vips::VImage::new_from_file(input_name);
    if (input.bands() != 4) {
        return 1; // 不是4band影像
    }

    input = input / 255.;
    auto input_zero = (input == 0).ifthenelse(0, 1);

    auto ndvi = (input[3] - input[0]) / (input[3] + input[0]);
    auto greater = (ndvi > ndvi_ratio).ifthenelse(1, 0);
    auto less = (ndvi <= ndvi_ratio).ifthenelse(1, 0);
    auto green_new = greater * (input[1] * green_ratio + input[3] * (1 - green_ratio)) + less * input[1];

    if (keep_nir == 4) {
        output = input[0].bandjoin(green_new).bandjoin(input[2]).bandjoin(input[3]);
    } else if (keep_nir == 3) {
        output = input[0].bandjoin(green_new).bandjoin(input[2]);
    } else
        return 1;

    if (output.bands() == 4) {
        output = output * input_zero;
    } else {
        output = output * (input_zero[0].bandjoin(input_zero[1]).bandjoin(input_zero[2]));
    }

    output = (output * 255).cast(VIPS_FORMAT_UCHAR);
    save(output, output_name);

    return 0;
}

vips::VImage zero_hist() {
    double a[256];

    for (int i = 0; i < 256; i++) {
        i == 0 ? a[i] = 0.0 : a[i] = 1.0;
    }

    return vips::VImage::new_matrix(256, 1, a, 256);
}

void histogram_match(VipsImage *input, const char *ref_name, VipsImage **output) {
    vips::VImage img, ref, in1, in2, dst;
    size_t data_size;
    void *buff;

    in1 = vips::VImage(input, vips::NOSTEAL);
    buff = in1.write_to_memory(&data_size);
    img = vips::VImage::new_from_memory_steal(buff, data_size, in1.width(), in1.height(), in1.bands(), VIPS_FORMAT_UCHAR);
    ref = vips::VImage::new_from_file(ref_name);

    vips::VImage out_bands[3], band4;
    double l[256];
    int lookup_value = 0;
    auto input_zero = (img == 0).ifthenelse(0, 1);

    if (img.bands() == 4) {
        band4 = img[3];
    }

    for (int i = 0; i < 3; i++) {
        auto img_hist = img[i].hist_find();
        auto ref_hist = ref[i].hist_find();
        auto temp = zero_hist();
        img_hist = img_hist * temp;
        ref_hist = ref_hist * temp;

        auto img_hist_cum = img_hist.hist_cum();
        auto ref_hist_cum = ref_hist.hist_cum();
        auto img_cdf = (img_hist_cum / img_hist_cum.max());
        auto ref_cdf = (ref_hist_cum / ref_hist_cum.max());

        // 构造LUT
        auto c = x_vips_to_arma_mat(img_cdf);
        auto d = x_vips_to_arma_mat(ref_cdf);
        for (int m = 0; m < 256; m++) {
            auto src_value = c[m];
            for (int n = 0; n < 256; n++) {
                auto ref_value = d[n];
                if (ref_value >= src_value) {
                    lookup_value = n;
                    break;
                }
            }
            l[m] = lookup_value;
        }

        auto lut = vips::VImage::new_matrix(256, 1, l, 256);
        out_bands[i] = (img[i]).maplut(lut);
    }
    if (img.bands() == 4) {
        dst = out_bands[0].bandjoin(out_bands[1]).bandjoin(out_bands[2]).bandjoin(band4);
        dst = dst * input_zero;
    } else {
        dst = dst * input_zero;
        dst = out_bands[0].bandjoin(out_bands[1]).bandjoin(out_bands[2]).bandjoin(255);
    }
    dst = dst.cast(VIPS_FORMAT_UCHAR);
    vips_copy(dst.get_image(), *(&output), NULL);
}

void histogram_match_batch(const char *input_name, const char *reference, const char *output_name) {
    auto img = vips::VImage::new_from_file(input_name);
    auto ref = vips::VImage::new_from_file(reference);

    vips::VImage out_bands[3], band4, dst;
    double l[256];
    int lookup_value = 0;
    auto input_zero = (img == 0).ifthenelse(0, 1);

    if (img.bands() == 4) {
        band4 = img[3];
    }

    for (int i = 0; i < 3; i++) {
        auto img_hist = img[i].hist_find();
        auto ref_hist = ref[i].hist_find();
        auto temp = zero_hist();
        img_hist = img_hist * temp;
        ref_hist = ref_hist * temp;

        auto img_hist_cum = img_hist.hist_cum();
        auto ref_hist_cum = ref_hist.hist_cum();
        auto img_cdf = (img_hist_cum / img_hist_cum.max());
        auto ref_cdf = (ref_hist_cum / ref_hist_cum.max());

        // 构造LUT
        auto c = x_vips_to_arma_mat(img_cdf);
        auto d = x_vips_to_arma_mat(ref_cdf);
        for (int m = 0; m < 256; m++) {
            auto src_value = c[m];
            for (int n = 0; n < 256; n++) {
                auto ref_value = d[n];
                if (ref_value >= src_value) {
                    lookup_value = n;
                    break;
                }
            }
            l[m] = lookup_value;
        }

        auto lut = vips::VImage::new_matrix(256, 1, l, 256);
        out_bands[i] = (img[i]).maplut(lut);
    }
    if (img.bands() == 4) {
        dst = out_bands[0].bandjoin(out_bands[1]).bandjoin(out_bands[2]).bandjoin(band4);
    } else {
        dst = out_bands[0].bandjoin(out_bands[1]).bandjoin(out_bands[2]);
    }

    dst = dst * input_zero;
    dst = dst.cast(VIPS_FORMAT_UCHAR);

    save(dst, output_name);
}