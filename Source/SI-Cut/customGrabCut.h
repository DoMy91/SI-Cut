//
//  customGrabCut.h
//  SI-Cut
//
//  Created by Domenico on 01/12/15.
//  Copyright (c) 2015 Domenico. All rights reserved.
//

#ifndef __SI_Cut__customGrabCut__
#define __SI_Cut__customGrabCut__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "gcgraph.hpp"

double calcBeta( const cv::Mat& img );
void calcNWeights( const cv::Mat& img, cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma );
void initMask(cv::Mat& mask,const cv::Mat& classification);
void constructGCGraph( const cv::Mat& img, const cv::Mat& mask,const cv::Mat& classification, double lambda,
                      const cv::Mat& leftW, const cv::Mat& upleftW, const cv::Mat& upW, const cv::Mat& uprightW,
                      GCGraph<double>& graph );
void estimateSegmentation( GCGraph<double>& graph, cv::Mat& mask );
void customGrabCut(const cv::Mat& image,const cv::Mat& classification,cv::Mat& result);

#endif /* defined(__SI_Cut__customGrabCut__) */
