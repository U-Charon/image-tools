//
// created by WangBo
//
#include "../include/xmap_util.hpp"

/*
基于arma::cube的卷积，使用灵活，兼容三种模式min, mean, max
Parameters:
    input_data: arma::cube, each_pixel∈[0, 1];
    kernel_size: {k_h, k_w}, 卷积核的大小，允许 h != w;
    stride: 1, 2, 3... 不能大于min{k_h, k_w};
    conv_mode: {0, 1, 2} --> {min_pooling, mean, max_pooling}
*/
arma::cube convolution_by_matrix(arma::cube input_data,
                                 const int kernel_size[2], const int stride, const int conv_model)
{
    // using namespace arma
    input_data = input_data;
    int h = input_data.n_cols; // 列数 即为 高
    int w = input_data.n_rows; // 行数 即为 宽
    int c = input_data.n_slices;
    int result_h, result_w;
    if (h % stride == 0)
    {
        result_h = h / stride;
    }
    else
    {
        result_h = h / stride + 1;
    }

    if (w % stride == 0)
    {
        result_w = w / stride;
    }
    else
    {
        result_w = w / stride + 1;
    }
    // printf("result_h: %d, result_w: %d\n", result_h, result_w);

    arma::cube conv_result(h, w, c);
    if (conv_model == 0)
    {
        conv_result = arma::ones(result_w, result_h, c);
    }
    else
    {
        conv_result = arma::zeros(result_w, result_h, c);
    }

    int padding_h, padding_w;
    padding_h = (result_h - 1) * stride + kernel_size[0] - h;
    padding_w = (result_w - 1) * stride + kernel_size[1] - w;

    int padding_top = padding_h / 2;
    int padding_bottom = padding_h - padding_top;
    int padding_left = padding_w / 2;
    int padding_right = padding_w - padding_left;
    // printf("top:%d, bottom:%d, left:%d, right:%d\n", padding_top, padding_bottom, padding_left, padding_right);

    cv::Mat input_data_cv = x_arma_to_cv(input_data);
    cv::Mat padding_data_cv;
    cv::copyMakeBorder(input_data_cv, padding_data_cv,
                       padding_top, padding_bottom, padding_left, padding_right,
                       cv::BORDER_REFLECT);
    arma::cube padding_data = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>)padding_data_cv);
    // padding_data.print("padding_data:");
    // printf("padding_data: h:%d, w:%d, c:%d\n", padding_data.n_cols, padding_data.n_rows, padding_data.n_slices);

    for (int i = 0; i < kernel_size[0]; i++)
    { // col
        arma::Col<arma::uword> col_index = arma::regspace<arma::Col<arma::uword>>(i, stride, i + h - 1);
        // col_index.print("col_index:");
        for (int j = 0; j < kernel_size[1]; j++)
        { // row
            arma::Col<arma::uword> row_index = arma::regspace<arma::Col<arma::uword>>(j, stride, j + w - 1);
            // row_index.print("row_index:");

            for (int k = 0; k < c; k++)  // 逐通道 取值
            {
                if (conv_model == 1)
                {
                    conv_result.slice(k) += padding_data.slice(k)(row_index, col_index);
                }
                if (conv_model == 0)
                {
                    conv_result.slice(k) = arma::min(conv_result.slice(k), padding_data.slice(k)(row_index, col_index));
                }
                if (conv_model == 2)
                {
                    conv_result.slice(k) = arma::max(conv_result.slice(k), padding_data.slice(k)(row_index, col_index));
                }
            }

            // if (conv_model == "mean")
            // {
            //     // printf("%d, %d, %d, %d", j, i, j + w-1, i + h -  1);
            //     conv_result += padding_data.tube(j, i, j + w-1, i + h-1);
            //     // conv_result.print("conv_result:");
            // }
            // if (conv_model == "min_pooling")
            // {
            //     conv_result = arma::min(conv_result, padding_data.tube(i, j, i + h - 1, j + w - 1));
            // }
            // if (conv_model == "max_pooling")
            // {
            //     conv_result = arma::max(conv_result, padding_data.tube(i, j, i + h - 1, j + w - 1));
            // }
        }
    }

    if (conv_model == 1)
    {
        conv_result = conv_result / (kernel_size[0] * kernel_size[1]);
    }
    return conv_result;
}

int main(int argc, char **argv)
{

    if (vips_init(argv[0]))
    {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);

    const char *f = R"(F:\data_analysis\Multiple_Auto-Adapting_Color_Balancing\img_balance_to_target\5m.tif)";
    auto input = vips::VImage::new_from_file(f);
    // printf("%d, %d\n", input.height(), input.width()); // 2064 3004

    cv::Mat input_cv = x_vips_to_cv_64f(input / 255);
    arma::cube input_arma = x_cv_to_arma_cube((cv::Mat_<cv::Vec<double, 3>>)input_cv);
    // printf("input_arma shape: %d, %d, %d\n", input_arma.n_cols, input_arma.n_rows, input_arma.n_slices); // 2064 3004
    // printf("input_arma min: %f, max: %f\n",
    //        x_arma_cube_stats(input_arma)[1],
    //        x_arma_cube_stats(input_arma)[2]);

    
    int k_size[2] = {3, 3};
    int s = 3;
    const int c_model = 0;

    auto start = x_get_current_ms();
    arma::cube conv_out = convolution_by_matrix(input_arma, k_size, s, c_model);
    auto end = x_get_current_ms();
    printf("total computation time %.8f second.\n", (double)(end - start) / 1000);

    auto x = x_arma_cube_stats(conv_out);
    // printf("%f, %f, %f\n", x[0], x[1], x[2]);

    auto conv_out_cv = x_arma_to_cv(conv_out);
    x_display_cv_image(conv_out_cv, "result", 0);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}
