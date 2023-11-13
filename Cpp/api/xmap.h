//
// Created by jerry on 2022/4/11.
//

#ifndef XMAPBALANCE_XMAP_H
#define XMAPBALANCE_XMAP_H

#pragma once

#include <vips/vips8>

#include <armadillo>
#include <opencv2/opencv.hpp>

extern "C" VipsImage *new_image_from_file(const char *filename);
extern "C" void new_image_from_file1(const char *filename, VipsImage **out);
//extern "C" VipsImage *coarse_light_remove(VipsImage *input, double offset, double sigmaD0);
extern "C" void coarse_light_remove(VipsImage *input, VipsImage **output, double offset, double sigmaD0);
//extern "C" VipsImage *pixel_balance(VipsImage *input, int blk_size);
extern "C" VipsImage *pixel_balance(VipsImage *input, int blk_size);
extern "C" VipsImage *pixel_balance1(VipsImage *input, int blk_size);
extern "C" VipsImage *radiation_correction(VipsImage *input, double min_value, double max_value, double t1, double gamma, double lambda_adjust);

extern "C" void optimized_stretch(VipsImage *input, VipsImage **output, double min_percent, double max_percent, double min_adjust, double max_adjust, int order);
extern "C" void optimized_stretch_batch_calculate(char *input_name, char *output_name, double min_adjust, double max_adjust,
                                                  int order, const int *low, const int *high);
extern "C" void optimized_stretch_batch_no_calculate(char *input_name, char *output_name, double min_percent, double max_percent,
                                                     double min_adjust, double max_adjust, int order);

extern "C" void sigmoid_stretch(VipsImage *input, VipsImage **output, double min, double max, double min_percent, double max_percent, double min_adjust,
                                double max_adjust, double alpha, double beta, int order);
extern "C" void sigmoid_stretch_batch_calculate(char *input_name, char *output_name, double min, double max,
                                                double min_adjust, double max_adjust, double alpha, double beta,
                                                int order, const int *low, const int *high);
extern "C" void sigmoid_stretch_batch_no_calculate(char *input_name, char *output_name, double min, double max,
                                                   double min_percent, double max_percent, double min_adjust,
                                                   double max_adjust, double alpha, double beta, int order);
extern "C" VipsImage *memory_to_from(VipsImage *input);
extern "C" VipsImage *hist_equalize(VipsImage *input);
extern "C" VipsImage *hist_equalize_clahe(VipsImage *input, double clip, int win);
extern "C" VipsImage *haze_remove(VipsImage *input, int radius, double ratio);
extern "C" VipsImage *double_filter(VipsImage *input, int radius, double sigmaColor, double sigmaSpace);

extern "C" void color_transfer(VipsImage *input, VipsImage *target, VipsImage **output, double p, double overlap, double alpha);

extern "C" void color_transfer_no_joint_batch(const char *base_name, const char *input_dir, const char *output_dir,
                                              float p, float overlap, float alpha);
extern "C" VipsImage *color_transfer_3order(VipsImage *input, VipsImage *target, double p, double overlap, double alpha);

extern "C" double vips_image_min(VipsImage *input);
extern "C" double vips_image_max(VipsImage *input);
extern "C" void save_tiff(VipsImage *input, const char *filename, int compression, int quality);
extern "C" void calculate_hist_total(const char *directory, int bands, int order, double min_p, double max_p, int *a, int *b);
extern "C" VipsImage *vips_agcwd(VipsImage *input, double alpha);
extern "C" void contrast_agcwd_batch(const char *input_name, const char *output_name, float alpha, int keep_nir);

extern "C" VipsImage *vips_aindane(VipsImage *input, int sigma1, int sigma2, int sigma3);
extern "C" VipsImage *vips_wthe(VipsImage *input, double r, double v);
extern "C" VipsImage *vips_agcie(VipsImage *input);

extern "C" void enhance_green_band(VipsImage *input, VipsImage **output, float ndvi_ratio, float green_ratio);
extern "C" int enhance_green_band_batch(const char *input_name, const char *output_name, float ndvi_ratio, float green_ratio, int keep_nir);

extern "C" void histogram_match(VipsImage *input, const char *ref_name, VipsImage **output);
extern "C" void histogram_match_batch(const char *input_name, const char *reference, const char *output_name);

#endif //XMAPBALANCE_XMAP_H
