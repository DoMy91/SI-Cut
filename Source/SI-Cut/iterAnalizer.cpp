//
//  iterAnalizer.cpp
//  SI-Cut
//
//  Created by Domenico on 19/12/15.
//  Copyright (c) 2015 Domenico. All rights reserved.
//

#include "iterAnalizer.h"
#include "structInconsist.h"
#include "customGrabCut.h"
#include <iostream>

using namespace cv;
using namespace std;

/*
 Calcolo della percentuale di variazione della regione T attuale rispetto a quella dell'iterazione precedente
 abs(T_prec-T)/abs(T_prec)
 */
void regionVariationPercent(const Mat& T,const Mat& T_prec,float& var_percent){
    Mat difference(T.rows,T.cols,CV_16S);
    subtract(T_prec, T, difference,noArray(),CV_16S);
    var_percent=(float)countNonZero(difference)/(float)countNonZero(T_prec);
}

/*
 Calcolo delle distanze ortogonali massime tra il rettangolo di input e la bounding-box
 della regione T attuale
 */
void closestSideDistances(const Mat& T,Point tl_mask,Point br_mask,int& hor_dist,int& ver_dist){
    Rect boundRect=boundingRect(T);
    hor_dist=max(abs(tl_mask.x-boundRect.tl().x),abs(br_mask.x-boundRect.br().x));
    ver_dist=max(abs(tl_mask.y-boundRect.tl().y),abs(br_mask.y-boundRect.br().y));
}

void bestInterval(vector<uchar> vtcK_list,int& t_fb,int& t_bb){
    vector<uchar> f;
    int t_h,t_m,v_bw,y,t_tmp;
    float f_tm,min_val=255,max_val=0;
    bilateralFilter(vtcK_list, f,9,10,10);
    cout<<"\nVTCK list"<<endl;
    for(int i=0;i<f.size();i++)
        cout<<(int)f[i]<<" ";
    cout<<endl;
    //Const area
    //------------------------------------------
    const float TH_FLAT=2;
    const int V_MAX=2;
    const float C=0.1;
    const int TH=7;
    //------------------------------------------
    
    //trovo t_h (l'indice dell'elemento massimo di f)
    for(int i=0;i<f.size();i++){
        if(f[i]>=max_val){
            max_val=f[i];
            t_h=i;
        }
    }
    f_tm=((float)f[t_h]+(float)f[1])/2;
    for(int i=0;i<f.size();i++){
        if(abs(f[i]-f_tm)<min_val){
            min_val=abs(f[i]-f_tm);
            t_m=i;
        }
    }
    //Se la pendenza della curva è eccessivamente ridotta t_h e t_h-1 sono assegnate a t_bb e t_fb
    if((t_h-1)==0 || (f[t_h]-f[1])/(t_h-1) <TH_FLAT){
        t_bb=t_h;
        t_fb=t_h-1;
        return;
    }
    //Backward search:cerco la prima iterazione al di sotto della retta tangente di t_h
    t_bb=t_h;
    v_bw=min(V_MAX,f[t_h]-f[t_h-1]); //pendenza locale della tangente in t_h
    y=f[t_h];                        //altezza della tangente all'iterazione t
    for(int t=t_h;t>=t_m;t--){
        if(f[t]+C<y){                //se la distanza tra f[t] e la retta tangente è più grande di C
            t_bb=t;
            break;
        }
        y-=v_bw;                     //aggiorno l'altezza della retta tangente
    }
    //Forward search:cerco la prima iterazione la cui derivata sinistra è molto più grande di quella destra
    t_fb=t_bb;
    for(int t=t_m+1;t<=t_h;t++){
        if((f[t]-f[t-1])>(f[t+1]-f[t]+TH)){
            t_fb=t;
            break;
        }
    }
    //Se t_fb>t_bb scambio i loro valori
    if(t_fb>t_bb){
        t_tmp=t_fb;
        t_fb=t_bb;
        t_bb=min(t_tmp,t_fb+1);
    }
}

/*
 Ottimizzazione finale del contorno
 */
void wideBandContourOptimization(const Mat& image,const Mat& mask,const Mat& F_fb,const Mat& F_bb,Mat& result){
    Mat Ebb=F_bb.clone();
    Mat Dfb=F_fb.clone();
    Mat classification=Mat::zeros(F_fb.rows,F_fb.cols,CV_8UC3);
    vector<vector<Point> > contours_Ebb,contours_Dfb;
    vector<Vec4i> hierarchy;
    const uchar *p_b,*p_f;
    int i,j;
    //Const area
    //------------------------------------------
    const int K_E=1;
    const int K_D=3;
    //------------------------------------------
    //effettuo K_E erosioni di F_bb
    for(i=0;i<K_E;i++)
        erode(Ebb,Ebb,Mat());
    //K_D dilatazioni di F_fb
    for(i=0;i<K_D;i++)
        dilate(Dfb, Dfb, Mat());
    findContours( Ebb.clone(), contours_Ebb, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( Dfb.clone(), contours_Dfb, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    /*
     Assegnazione delle probabilità per l'ottimizzazione del contorno con graph-cut:
     i pixel appartenenti a F_bb sono di probabile foreground,
     i pixel appartenenti a F_fb-F_bb sono neutrali
     i pixel rimanenti appartenenti al rettangolo di input sono di probabile background
     i pixel esterni al rettangolo di input sono di sicuro background
     */
    for(i=0;i<F_fb.rows;i++){
        p_b=F_bb.ptr<uchar>(i);
        p_f=F_fb.ptr<uchar>(i);
        for(j=0;j<F_fb.cols;j++){
            if(mask.at<uchar>(i,j)==255){
                if(F_bb.at<uchar>(i,j)==255) //probable foreground
                    classification.at<Vec3b>(i, j)[1]=75;
                else if(Dfb.at<uchar>(i,j)==255){ //neutral
                    classification.at<Vec3b>(i, j)[1]=50;
                    classification.at<Vec3b>(i, j)[2]=50;
                }
                else //probable background
                    classification.at<Vec3b>(i, j)[2]=75;
            }
            else //definite background
                classification.at<Vec3b>(i, j)[2]=100;
        }
    }
    Scalar definiteFg = Scalar( 0,100,0 ); //green for foreground
    Scalar definiteBg = Scalar( 0,0,100 ); //red for background
    //i pixel appartenenti al contorno di Ebb sono di sicuro foreground
    for( int i = 0; i< contours_Ebb.size(); i++ )
        drawContours( classification, contours_Ebb, i, definiteFg, 2, 8, hierarchy, 0, Point() );
    //i pixel appartenenti al contorno di Dfb sono di sicuro background
    for( int i = 0; i< contours_Dfb.size(); i++ )
        drawContours( classification, contours_Dfb, i, definiteBg, 2, 8, hierarchy, 0, Point() );
    imshow("classification wide band", classification);
    imwrite("classification_wide_band.png",classification);
    waitKey();
    customGrabCut(image, classification, result); //graph-cut
    showImage(image, result);
}


