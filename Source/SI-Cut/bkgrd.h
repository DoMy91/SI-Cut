//
//  bkgrd.h
//  SI-Cut
//
//  Created by Domenico on 26/11/15.
//  Copyright (c) 2015 Domenico. All rights reserved.
//

#ifndef SI_Cut_bkgrd_h
#define SI_Cut_bkgrd_h

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

void ind2sub(int ind, int& x, int& y);
int shift2lab(int dx, int dy);
void lab2shift(int lab, int& dx, int& dy);
void computeGradient(const cv::Mat& image,cv::Mat& gradX,cv::Mat& gradY);
bool isInside(int p_x,int p_y);
void threesholdMask(cv::Mat& mask);
void interpNearestShift(cv::Mat& shiftX,cv::Mat& shiftY,cv::Mat& mask);
float LocationPenalty(cv::Point p,cv::Point r_P,float width,float height);
float dataFn(int site,int label,void *d);
float SmoothCostFn(int s1,int s2,int l1,int l2,void *d);
void gridGraphIndividually(int num_labels,const cv::Mat& image,const cv::Mat& mask,const cv::Mat& Vreg,const cv::Mat& gradX,const cv::Mat& gradY,cv::Mat& shiftX,cv::Mat& shiftY);
void backgroundSPrediction(const cv::Mat& image,const cv::Mat& mask,const cv::Mat& Vregion,cv::Mat& background,bool flag);
bool computeShift(int p_x,int p_y,int label,const cv::Mat& shiftX,const cv::Mat& shiftY,const cv::Mat& mask,int& sp_x,int& sp_y);
#endif
