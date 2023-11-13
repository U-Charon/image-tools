// #include <vips/vips8>
#include "../include/xmap_util.hpp"

/*
 * 增强中间值域[x1, x2]的对比度, 压缩[0, x1]以及[x2, 1]的对比度;
 * beta值越大, [x1, x2]的范围越小;
 * out: 值域范围(0, 1);
 */
vips::VImage sigmiod_stretch_1(const vips::VImage &input_data, double beta)
{
    std::vector<vips::VImage> results;
    for (int i = 0; i < input_data.bands(); i++)
    {
        auto meanv = input_data[i].avg();
        auto x = 1 / (1 + (beta * (meanv - input_data[i])).exp());
        // auto x = 1 / (1 + (beta * (alpha - input_data[i])).exp());
        results.push_back(x);
        // results.push_back((x-x.min())/(x.max()-x.min()));
    }
    vips::VImage out = results[0].bandjoin(results[1]).bandjoin(results[2]);
    return out;
}

/*
 * 增强中间值域[x1, x2]的对比度, 压缩[0, x1]以及[x2, 1]的对比度;
 * beta: 值越大, [x1, x2]的范围越小;
 * out: 值域范围是[0, 1];
 */
vips::VImage sigmiod_stretch_2(const vips::VImage &input_data, double beta)
{
    std::vector<vips::VImage> results;
    for (int i = 0; i < input_data.bands(); i++)
    {
        auto u = (input_data[i] - input_data[i].min()) / (input_data[i].max() - input_data[i].min());
        auto meanv = u.avg();
        auto x = 1 / (1 + (beta * (meanv - u)).exp());
        auto a = 1 / (1 + exp((beta * (meanv - 0))));
        auto b = 1 / (1 + exp((beta * (meanv - 1))));
        results.push_back((x - a) / (b - a));  // 等价于 (x-x.min())/(x.max()-x.min())
    }
    auto out = results[0].bandjoin(results[1]).bandjoin(results[2]);
    return out;
}

/*
 * input_data 值域范围必须是 [0, 255];
 * out: [0, 1]
 */
vips::VImage sigmiod_stretch_3(const vips::VImage &input_data, double beta)
{
    auto u = (beta / 255.) * input_data - (beta / 2); // 对于input_data[0, 255] ----> u[0, beta]-beta/2 ----> [-beta/2, beta/2]
    std::vector<vips::VImage> results;
    for (int i = 0; i < input_data.bands(); i++)
    {
        auto meanv = u[i].avg();  // meanv: mid
        auto a = (u[i] - meanv).exp();
        auto b = (meanv - u[i]).exp();
        auto sigm = (a - b) / (a + b); // sigm ∈ [-1, 1]  本质雨stretch_1, streth_2有区别；
        results.push_back((sigm + 1) / 2);
    }
    auto out = results[0].bandjoin(results[1]).bandjoin(results[2]);
    return out;
}

int main(int argc, char **argv)
{
    if (vips_init(argv[0]))
    {
        vips_error_exit("unable to init vips");
    }
    vips_leak_set(TRUE);

    const char *img_path = R"(F:\data_analysis\Optimized_Linear_Stretch\08bit3channel_out\cs1.tif)";
    // const char *img_path = R"(F:\data_analysis\illumination_contrast_balancing\Correction\test1.png)";
    // const char *img_path = R"(F:\data_analysis\Optimized_Linear_Stretch\16bit4channel_tif_out2\hanshou_zy3.tif)";
    // const char *img_path = R"(F:\data_analysis\Optimized_Linear_Stretch\16bit4channel_tif\TEST1.tif)";

    auto in_data = vips::VImage::new_from_file(img_path);
    // in_data = in_data[2].bandjoin(in_data[1]).bandjoin(in_data[0]);

    // auto out_data = sigmiod_stretch_1(in_data/255, 6);  
    // auto out_data = sigmiod_stretch_2(in_data, 6);
    auto out_data = sigmiod_stretch_3(in_data, 4.);
    x_display_vips_image((out_data * 255).cast(VIPS_FORMAT_UCHAR), "out", 0);
    cv::waitKey();
    cv::destroyAllWindows();
}
