//
//  customGrabCut.cpp
//  SI-Cut
//
//  Created by Domenico on 01/12/15.
//  Copyright (c) 2015 Domenico. All rights reserved.
//

#include "customGrabCut.h"
#include <vector>
using namespace cv;
using namespace std;

/*
 Calculate beta - parameter of GrabCut algorithm.
 beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
 */
double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );
    
    return beta;
}

/*
 Calculate weights of noterminal vertices of graph.
 beta and gamma - parameters of GrabCut algorithm.
 */
void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

/*
 Inizializzo la maschera per la costruzione del grafo del grabCut.
 GC_BGD:Pixel di sicuro background
 GC_PR_BGD:Pixel di probabile background
 GC_PR_FGD:Pixel di probabile foreground
 GC_FGD:Pixel di sicuro foreground
 */
void initMask(Mat& mask,const Mat& classification){
    const Vec3b *p;
    for(int i=0;i<classification.rows;i++){
        p=classification.ptr<Vec3b>(i);
        for(int j=0;j<classification.cols;j++){
            if(p[j][2]!=0){  //background
                if(p[j][2]==100)
                    mask.at<uchar>(i, j)=GC_BGD;
                else
                    mask.at<uchar>(i, j)=GC_PR_BGD;
            }
            else{  //foreground
                if(p[j][1]==100)
                    mask.at<uchar>(i, j)=GC_FGD;
                else
                    mask.at<uchar>(i, j)=GC_PR_FGD;
            }
        }
    }
}

/*
 Construct GCGraph
 */
void constructGCGraph( const Mat& img, const Mat& mask,const Mat& classification, double lambda,
                      const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                      GCGraph<double>& graph )
{
    int vtxCount = img.cols*img.rows,
    edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
    float prob;
    graph.create(vtxCount, edgeCount);
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);
            
            // set t-weights
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD )
            {
                prob=(float)classification.at<Vec3b>(p)[2]/100; //red background
                fromSource = -log( prob );
                toSink = -log( 1-prob );
            }
            else if(mask.at<uchar>(p) == GC_PR_FGD){
                prob=(float)classification.at<Vec3b>(p)[1]/100; //green foreground
                fromSource =-log( 1-prob );
                toSink = -log( prob );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights( vtxIdx, fromSource, toSink );
            
            // set n-weights
            if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}

/*
 Estimate segmentation using MaxFlow algorithm
 */
void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    graph.maxFlow();
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

/*
 Versione modificata del grabCut.Riceve in input un'immagine e la classificazione dei suoi pixel
 con relative probabilità (immagine a colori dove i pixel di background sono rossi e quelli di
 foreground verdi,con intensità pari alla probabilità di background/foreground).Costruisce un grafo
 rappresentante i pixel dell'immagine settando come data-term tali probabilità (invece di GMM) e
 calcola il taglio minimale restituendo in output il labeling prodotto (immagine binaria dove i pixel
 neri sono i pixel di background e quelli bianchi di foreground).
 */
void customGrabCut(const Mat& image,const Mat& classification,Mat& result){
    Mat mask(image.rows,image.cols,CV_8U);
    uchar *p;
    initMask(mask,classification);
    const double gamma = 50;
    const double lambda = 9*gamma;
    const double beta = calcBeta( image );
    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( image, leftW, upleftW, upW, uprightW, beta, gamma );
    GCGraph<double> graph;
    constructGCGraph(image, mask,classification,lambda, leftW, upleftW, upW, uprightW, graph );
    estimateSegmentation( graph, mask );
    for(int i=0;i<mask.rows;i++){
        p=mask.ptr<uchar>(i);
        for(int j=0;j<mask.cols;j++){
            if(p[j]==GC_PR_BGD || p[j]==GC_BGD)
                result.at<uchar>(i,j)=0;
            else
                result.at<uchar>(i, j)=255;
        }
    }
}
