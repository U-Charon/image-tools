//
// Created by jerry on 2022/5/3.
//
#include <vips/vips8>

//#define ARMA_DONT_USE_BLAS
//#define ARMA_DONT_USE_LAPACK
//#define ARMA_DONT_USE_WRAPPER

#include <armadillo>
#include <opencv2/opencv.hpp>

vips::VImage matToVipsDouble(const cv::Mat &mat) {
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

template<typename T, int NC>
arma::Cube<T> cv_to_arma(const cv::Mat_<cv::Vec<T, NC>> &src) {
    std::vector<cv::Mat_<T>> channels;
    arma::Cube<T> dst(src.cols, src.rows, NC);
    for (int c = 0; c < NC; ++c)
        channels.emplace_back(src.rows, src.cols, dst.slice(c).memptr());
    cv::split(src, channels);
    return dst;
}

template<typename T>
cv::Mat arma_to_cv(const arma::Cube<T> &src) {
    std::vector<cv::Mat_<T>> channels;
    for (size_t c = 0; c < src.n_slices; ++c) {
        auto *data = const_cast<T *>(src.slice(c).memptr());
        channels.emplace_back(int(src.n_cols), int(src.n_rows), data);
    }
    cv::Mat dst;
    cv::merge(channels, dst);
    return dst;
}

template<typename T>
cv::Mat arma_to_cv_single(const arma::Mat<T> &src) {

    return {static_cast<int>(src.n_cols), static_cast<int>(src.n_rows), CV_8UC1, (T *) src.memptr()};
}

cv::Mat vipsToMatRGB(const vips::VImage &in) {
    size_t dataSize;
    void *data = in.write_to_memory(&dataSize);
    cv::Mat mat = cv::Mat(in.height(), in.width(), CV_8UC3, data);
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    return mat;
}

cv::Mat vipsToMatRGBDOUBLE(const vips::VImage &in) {
    size_t dataSize;
    void *data = in.write_to_memory(&dataSize);
    cv::Mat mat = cv::Mat(in.height(), in.width(), CV_32F, data);
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    return mat;
}

void displayImage(const vips::VImage &in, const std::string &name, const int win_num) {
    if (in.bands() == 3) {
//        printf("using RGB mode to display image.\n");
        cv::Mat img = vipsToMatRGB(in);
        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::imshow(name, img);
        cv::moveWindow(name, (win_num * in.width() + 10), 20);
//        cv::waitKey(0);
//        cv::destroyWindow(name);
//        cv::imwrite("/Users/Jerry/dev/CLionProjects/xmapImageBalance/img/output.png", img);
    }
}

long long get_current_ms() {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );

    return ms.count();
}

void displayCVmat(const cv::Mat &in, const std::string &name, const int win_num) {
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, in);
    cv::moveWindow(name, (win_num * in.cols + 10), 20);
}

cv::Mat padding(const vips::VImage &in, int pad_h, int pad_w) {
    auto src = vipsToMatRGB(in);
    cv::Mat padded;

    cv::copyMakeBorder(src, padded, pad_h, pad_h, pad_w, pad_w,
                       cv::BORDER_REFLECT);
    padded.convertTo(padded, CV_64F);

    return padded;
}

cv::Mat padding(const vips::VImage &in, int top, int bottom, int left, int right) {
    auto src = vipsToMatRGB(in);
    cv::Mat padded;

    cv::copyMakeBorder(src, padded, top, bottom, left, right,
                       cv::BORDER_REFLECT);
    padded.convertTo(padded, CV_64F);

    return padded;
}

vips::VImage ones(int rows, int cols) {
    return (vips::VImage::black(rows, cols) + 1.0);
}

void test(const vips::VImage &input) {


    auto cv_input = vipsToMatRGB(input);

    auto Q = cv_to_arma((cv::Mat_<cv::Vec<uchar, 3>>) cv_input);
    printf("Q row=%llu, col=%llu\n", Q.n_rows, Q.n_cols);
    Q.tube(0, 0, 100, 50).fill(0);
    auto back = arma_to_cv(Q);
    auto r = arma_to_cv_single(Q.slice(0));


    displayCVmat(back, "a", 1);
    displayCVmat(r, "r", 2);
    cv::waitKey();
    cv::destroyAllWindows();
}

vips::VImage local_mean_function(const vips::VImage &in, double p, double overlap) {
    using namespace arma;

    const double constant = 128.0 / 45;  // 理想状态下的mean/std
    auto h = in.height();
    auto w = in.width();
    auto im_mean = in.stats()(4, 0)[0];
    auto im_std = in.stats()(5, 0)[0];

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
//    printf("adw_size=%d, adw_stride=%d, num_h=%d, num_w=%d, padding_h=%d, padding_w=%d, padding_top=%d, padding_bottom=%d, padding_left=%d, padding_right=%d\n",
//           adw_size, adw_stride, num_h, num_w, padding_h, padding_w, padding_top, padding_bottom, padding_left, padding_right);

    auto img_padding = padding(in, padding_top, padding_bottom, padding_left, padding_right);
//    printf("img padding shape [%d,%d,%d]\n", img_padding.rows, img_padding.cols, img_padding.channels());
    auto arma_padding = cv_to_arma((cv::Mat_<cv::Vec<double, 3>>) img_padding);


    arma_padding = arma_padding / 255.;
//    printf("arma padding shape [%d,%d,%d], %f, %f\n", arma_padding.n_cols, arma_padding.n_rows, arma_padding.n_slices, arma_padding.min(), arma_padding.max());
    cube local_mean_map(num_w, num_h, 3); //[152, 105, 3]
    for (int m = 0; m < num_h; m++) {
        auto adw_top = m * adw_stride;
        auto adw_bottom = adw_top + adw_size;
        for (int n = 0; n < num_w; n++) {
            auto adw_left = n * adw_stride;
            auto adw_right = adw_left + adw_size;
//            printf("i=%d, j=%d, left=%d, top=%d, right=%d, bottom=%d\n", m, n, adw_left, adw_top, adw_right, adw_bottom);
            auto temp = mean(arma_padding.tube(adw_left, adw_top, adw_right - 1, adw_bottom - 1));
//            mean(temp, 1).print("");
            local_mean_map.tube(n, m) = mean(temp, 1);
        }
    }
//    printf("-----------------%f, %f\n", local_mean_map.min(), local_mean_map.max());

    auto m_h_ = (num_h - 1) * adw_stride;
//    auto m_w_ = (num_w - 1) * adw_stride;
    auto m_w_ = (int) arma_padding.n_rows - (adw_size - 1);
//    auto m_h_ = (int) arma_padding.n_cols - (adw_size - 1);
    auto top_x = int((m_w_ - w) / 2);
    auto top_y = int((m_h_ - h) / 2);
//    printf("topx=%d, topy=%d, num_h=%d, num_w=%d\n", top_x, top_y, m_h_, m_w_);
    auto idx = vips::VImage::xyz(m_w_, m_h_);
    idx = idx / adw_stride;
    idx = idx.extract_area(top_x, top_y, w, h);

    auto cv_lmm = arma_to_cv(local_mean_map);
    auto result = matToVipsDouble(cv_lmm).mapim(idx);
//    printf("result min=%.8f, max=%.8f, mean=%.8f\n", result.min(), result.max(), result.avg());

    return result;
}

int main(int argc, char **argv) {
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\5m.tif)";
//    const char *t = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\5m_target1.tif)";
    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\chengdu.tif)";
    const char *t = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\chengdu_target3.tif)";
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\test1.png)";
//    const char *f = R"(C:\Users\zengy\CLionProjects\xmapBalance\img\5m_local_mean.png)";
    if (vips_init(argv[0])) {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);
    auto input = vips::VImage::new_from_file(f);
    auto target = vips::VImage::new_from_file(t);

    auto scale_opt = vips::VImage::option()->set("vscale", (double) input.height() / target.height());
    scale_opt->set("kernel", VIPS_KERNEL_LANCZOS3);
    target = target.resize((double) input.width() / target.width(), scale_opt);
    printf("target wh=[%d, %d]\n", target.width(), target.height());
//    target = target / 255.;
//    test(input);
    auto start = get_current_ms();
//    auto test = local_mean_function(input, 0.01, 0.2);
    auto result = local_mean_function(input, 0.05, 0.2);
    auto target_result = local_mean_function(target, 0.005, 0.2);


    // start compute final result
//    auto b0 = target.stats()(4, 1)[0] * ones(input.width(), input.height());
//    auto b1 = target.stats()(4, 2)[0] * ones(input.width(), input.height());
//    auto b2 = target.stats()(4, 3)[0] * ones(input.width(), input.height());
//    auto mean = b0.bandjoin(b1).bandjoin(b2);
//    printf("==========mean minmax, avg [%.10f, %.10f, %.10f]\n", mean.min(), mean.max(), mean.avg());

//    auto log_mean = mean.log();
    auto log_mean = target_result.log();
    auto log_result = result.log();
//    printf("log_mean min=%.10f, max=%.10f, log_result min=%.10f, max=%.10f\n", log_mean.min(), log_mean.max(), log_result.min(), log_result.max());

    auto gamma = log_mean / log_result;
//    printf("gamma min=%.10f, max=%.10f, avg=%.10f\n", gamma.min(), gamma.max(), gamma.avg());

    auto out = 1.0 * (input / 255.).pow(gamma);
    auto end = get_current_ms();
    printf("total computation time %.6f second.\n", (double) (end - start) / 1000);
    (out * 255).cast(VIPS_FORMAT_UCHAR).tiffsave(R"(C:\Users\zengy\CLionProjects\xmapBalance\img\chengdu_result1.tif)");

    displayImage((out * 255).cast(VIPS_FORMAT_UCHAR), "r", 0);
//    displayCVmat(test, "test", 1);
    cv::waitKey();
    cv::destroyAllWindows();


}