//
//  iterAnalizer.h
//  SI-Cut
//
//  Created by Domenico on 19/12/15.
//  Copyright (c) 2015 Domenico. All rights reserved.
//

#ifndef __SI_Cut__iterAnalizer__
#define __SI_Cut__iterAnalizer__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

void regionVariationPercent(const cv::Mat& T,const cv::Mat& T_prec,float& var_percent);
void closestSideDistances(const cv::Mat& T,cv::Point tl_mask,cv::Point br_mask,int& hor_dist,int& ver_dist);
void wideBandContourOptimization(const cv::Mat& image,const cv::Mat& mask,const cv::Mat& F_fb,const cv::Mat& F_bb,cv::Mat& result);
void bestInterval(std::vector<uchar> vtcK_list,int& t_fb,int& t_bb);

#endif /* defined(__SI_Cut__iterAnalizer__) */
