//
//  structInconsist.h
//  SI-Cut
//
//  Created by Domenico on 04/12/15.
//  Copyright (c) 2015 Domenico. All rights reserved.
//

#ifndef __SI_Cut__structInconsist__
#define __SI_Cut__structInconsist__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

extern cv::Rect maskR;
void structInconsAnalysis(const cv::Mat& image,const cv::Mat& background,const cv::Mat& mask,int& T,cv::Mat& result);
void inconsistencyValues(const cv::Mat& image,const cv::Mat& background,cv::Mat& inconsist);
void thresholdForeground(const cv::Mat& image,const cv::Mat& inconsistBg,const cv::Mat& inconsistFg,cv::Mat& classificationBN);
void narrowGraphRefinement(const cv::Mat& image,cv::Mat& classificationBN);
void pixelClassificationMap(const cv::Mat& inconsist,const cv::Mat& foregroundMap,const cv::Mat& mask,int T,cv::Mat& classification);
void pixelClassificationThreshold(const cv::Mat& inconsist,int T,cv::Mat& classificationBN);
void computeThreshold(const cv::Mat& inconsist,const cv::Mat& mask,float topPercentage,int& T);
void referenceReachableFiltering(cv::Mat& binImg);
void regionSizeFiltering(cv::Mat& binImg);
void showImage(const cv::Mat& image,const cv::Mat& result);
void findRectVertex(const cv::Mat& mask,cv::Point& tl,cv::Point& br);

#endif /* defined(__SI_Cut__structInconsist__) */
