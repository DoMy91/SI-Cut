//
//  structInconsist.cpp
//  SI-Cut
//
//  Created by Domenico on 04/12/15.
//  Copyright (c) 2015 Domenico. All rights reserved.
//

#include "structInconsist.h"
#include "bkgrd.h"
#include "customGrabCut.h"
#include <iostream>
#include <math.h> 

using namespace std;
using namespace cv;

/*
 STRUCTURAL INCONSISTENCY ANALYSIS
 */
void structInconsAnalysis(const Mat& image,const Mat& background,const Mat& T,int& vtcK,Mat& result){
    Mat inconsistBg=Mat::zeros(image.rows,image.cols,CV_8U);
    Mat inconsistFg(image.rows,image.cols,CV_8U);
    Mat classification=Mat::zeros(image.rows,image.cols,image.type());
    Mat classificationBN=Mat::zeros(image.rows,image.cols,CV_8U);
    Mat Vregion=Mat::zeros(image.rows,image.cols,CV_8U);
    Mat fP(maskR.height,maskR.width,image.type());
    Mat foregroundPrediction=Mat::zeros(image.rows, image.cols, image.type());
    float topPercentage=0.2;//top 20%
    inconsistencyValues(image, background,inconsistBg);
    computeThreshold(inconsistBg, T, topPercentage, vtcK);
    pixelClassificationThreshold(inconsistBg, vtcK, classificationBN);
    regionSizeFiltering(classificationBN);
    referenceReachableFiltering(classificationBN);
    bitwise_not(classificationBN, classificationBN); //mask inversion
    cout<<"\nREVERSE FOREGROUND PREDICTION"<<endl;
    backgroundSPrediction(image(maskR),classificationBN(maskR),Vregion(maskR),fP,false);
    fP.copyTo(foregroundPrediction(maskR));
    inconsistencyValues(image, foregroundPrediction,inconsistFg);
    thresholdForeground(image,inconsistBg, inconsistFg, classificationBN);
    pixelClassificationMap(inconsistBg,classificationBN,T,vtcK,classification);
    customGrabCut(image, classification,result);
    imshow("Graph-cut",result);
    waitKey();
    showImage(image, result);
}


/*
 Tale procedura prende in input un'immagine e la sua stima di background
 e ritorna in output l'immagine in scala di grigio delle inconsistenze.
 Valori alti rappresentano alte inconsistenze con il background (quindi probabili pixel di foreground)
 viceversa valori bassi rappresentano basse inconsistenze e quindi probabili pixel di background.
 */

void inconsistencyValues(const Mat& image,const Mat& background,Mat& inconsist){
    const Vec3b *p,*q;
    uchar *r;
    for(int i=maskR.y;i<maskR.y+maskR.height;i++){
        p=image.ptr<Vec3b>(i);
        q=background.ptr<Vec3b>(i);
        r=inconsist.ptr<uchar>(i);
        for(int j=maskR.x;j<maskR.x+maskR.width;j++){
            //e' necessario utilizzare il cast a (Scalar) in quanto se p[j]<q[j],senza utilizzare il cast ho che (p[j]-q[j]) viene troncato a 0
            r[j]=norm((Scalar)p[j]-(Scalar)q[j])*255/norm(Scalar(255,255,255)); //scalatura intervallo [0-255]
        }
    }
    GaussianBlur(inconsist, inconsist, Size(3,3),0,0,BORDER_DEFAULT);
    imshow("Inconsistency values", inconsist(maskR));
    waitKey();
}

/*
 Calcolo della soglia.L'obiettivo è di sogliare i pixel dell'immagine delle inconsistenze considerando
 pixel di background tutti quelli aventi un valore di inconsistenza <vtcK e pixel di foreground tutti quelli rimanenti.
 Prende in input la percentuale (k) di pixel sul totale che devono essere considerati di background,costruisce l'istogramma
 e calcola un valore vtcK in modo tale che il numero dei pixel all'interno della regione T,aventi valore di inconsistenza<=vtcK
 sia <= del k% dei pixel dell'immagine.
 */

void computeThreshold(const Mat& inconsist,const Mat& T,float topPercentage,int& vtcK){
    int i,j;
    float cumSum=0;
    vector<int> countBins(256,0);
    const uchar *p;
    for(i=maskR.y;i<maskR.y+maskR.height;i++){
        p=inconsist.ptr<uchar>(i);
        for(j=maskR.x;j<maskR.x+maskR.width;j++){
            if(T.at<uchar>(i,j)!=0)  //T region
                countBins[p[j]]++; //conto il numero di pixel appartenente ad ogni bin dell'istogramma
        }
    }
    i=0;
    while (cumSum<=topPercentage) {
        cumSum+=(float)countBins[i++]/countNonZero(T);
    }
    vtcK=i-1;
}


/*
 Costruisco un'immagine binaria dei candidati di background (0) e foreground (255) in base alla soglia T
 */

void pixelClassificationThreshold(const Mat& inconsist,int vtcK,Mat& classificationBN){
    //opencv tratta le immagini a colori secondo lo schema BGR.Quindi Vec3b[0]=blue,Vec3b[1]=green,Vec3b[2]=red
    const uchar *p;
    classificationBN=Mat::zeros(inconsist.rows,inconsist.cols,CV_8U);
    for(int i=maskR.y;i<maskR.y+maskR.height;i++){
        p=inconsist.ptr<uchar>(i);
        for(int j=maskR.x;j<maskR.x+maskR.width;j++){
            if(p[j]>vtcK)
                classificationBN.at<uchar>(i, j)=255;
        }
    }
    imshow("Per-pixel classification binarized", classificationBN(maskR));
    waitKey();
}


/*
 Tale procedura prende in input un'immagine binaria rappresentante i candidati di background (pixel neri) e quelli di foreground
 (pixel bianchi) e rimuove tutte le regioni connesse (inglobandole nel background) aventi un numero di pixel minore del 20% del numero di pixel
 della regione connessa più grande ad esclusione del background.
 */

void regionSizeFiltering(Mat& binImg){
    Mat labels;
    int numLabels,*ptr,i,j,max;
    uchar *p;
    /*
     computes the connected components labeled image of boolean image image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0 represents the background label.
     */
    numLabels=connectedComponents(binImg, labels);
    vector<int> countLabels(numLabels,0);
    //conto il numero di pixel appartenenti ad ogni label (regione connessa)
    for(i=maskR.y;i<maskR.y+maskR.height;i++){
        ptr=labels.ptr<int>(i);
        for(j=maskR.x;j<maskR.x+maskR.width;j++){
            countLabels[ptr[j]]++;
        }
    }
    max=*(max_element(countLabels.begin()+1, countLabels.end()));
    for(i=maskR.y;i<maskR.y+maskR.height;i++){
        p=binImg.ptr<uchar>(i);
        ptr=labels.ptr<int>(i);
        for(j=maskR.x;j<maskR.x+maskR.width;j++){
            if(p[j]!=0 && countLabels[ptr[j]]<max*20/100)
                p[j]=0;
        }
    }
    imshow("Region-size filtering",binImg(maskR));
    waitKey();
}


/*
 Tale procedura prende in input un'immagine binaria rappresentante i candidati di background (pixel neri) e quelli di foreground
 (pixel bianchi) e ingloba nel foreground tutti i pixel neri che non sono raggiungibili a partire dal bordo del rettangolo.
 */

void referenceReachableFiltering(Mat& binImg){
    Mat floodImg=binImg.clone();
    uchar *p,*q;
    /*
     La function floodFill() prende in input un'immagine,un punto ed un colore (bianco).Viene assegnato tale colore a tutti i pixel
     che sono raggiungibili a partire dal punto indicato.Quindi i pixel che rimangono neri devono diventare bianchi nell'immagine
     binaria dei candidati di foreground.
     */
    floodFill(floodImg, Point(0,0), Scalar(255));
    imshow("floodfill", floodImg(maskR));
    waitKey();
    for(int i=maskR.y;i<maskR.y+maskR.height;i++){
        p=binImg.ptr<uchar>(i);
        q=floodImg.ptr<uchar>(i);
        for(int j=maskR.x;j<maskR.x+maskR.width;j++){
            if(q[j]==0)
                p[j]=255;
        }
    }
    imshow("Reference-reachable filtering",binImg(maskR));
    waitKey();
}


/*
 Rifinisco il contorno dell'immagine binaria dei candidati di foreground effettuando erosione e dilatazione della maschera e
 considerando i pixel appartenenti all'intersezione delle due maschere come pixel neutrali (50% probabilità di background,50% probabilità
 di foreground) da stimare con min graph-cut.In tal modo il contorno viene rifinito in modo da adattarsi ai colori locali.
 */

void narrowGraphRefinement(const Mat& image,Mat& classificationBN){ //foreground candidates map
    Mat erosionDst,dilationDst;
    Mat classification=Mat::zeros(classificationBN.rows,classificationBN.cols,image.type());
    uchar *p,*q;
    erode(classificationBN,erosionDst,Mat());
    dilate(classificationBN, dilationDst, Mat());
    for(int i=0;i<erosionDst.rows;i++){
        p=erosionDst.ptr<uchar>(i);
        q=dilationDst.ptr<uchar>(i);
        for(int j=0;j<erosionDst.cols;j++){
            if(p[j]==255){ //definite foreground
                classification.at<Vec3b>(i,j)[1]=100;
            }
            else if(q[j]==255){ //neutral
                classification.at<Vec3b>(i,j)[1]=50;
                classification.at<Vec3b>(i,j)[2]=50;
            }
            else{ //definite background
                classification.at<Vec3b>(i,j)[2]=100;
            }
        }
    }
    customGrabCut(image, classification,classificationBN);
    imshow("Graph-cut",classificationBN(maskR));
    waitKey();
}


/*
 Visualizzo il contorno del foreground sull'immagine
 */
void showImage(const Mat& image,const Mat& result){
    Mat imageClone=image.clone();
    Mat resultClone=result.clone();
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( resultClone, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    Scalar color = Scalar( 0,255,0 );
    for( int i = 0; i< contours.size(); i++ )
        drawContours( imageClone, contours, i, color, 2, 8, hierarchy, 0, Point() );
    imshow( "Result", imageClone );
    waitKey(0);
}


/*
 Effettuo il mapping delle probabilità di background/foreground a partire dall'immagine delle inconsistenze,utilizzando
 la soglia vtcK (top k%) e una funzione gaussiana a media 0.
 */

void pixelClassificationMap(const Mat& inconsist,const Mat& foregroundMap,const Mat& T,int vtcK,Mat& classification){
    const uchar *p,*q;
    int pr;
    double y=log(0.5)/pow(vtcK,2);
    classification=Mat::zeros(inconsist.rows,inconsist.cols,CV_8UC3);
    for(int i=0;i<inconsist.rows;i++){
        p=inconsist.ptr<uchar>(i);
        q=foregroundMap.ptr<uchar>(i);
        for(int j=0;j<inconsist.cols;j++){
            if(T.at<uchar>(i,j)==0) //sure background
                classification.at<Vec3b>(i,j)[2]=100;
            else{
                pr=100*pow(M_E,y*pow(p[j],2)); //gaussian function mapping background probability
                if(q[j]==0)  //probable background
                    classification.at<Vec3b>(i,j)[2]=pr;
                else
                    classification.at<Vec3b>(i,j)[1]=100-pr;
            }
        }
    }
    imshow("Per-pixel classification", classification);
    waitKey();
}



/*
 Costruisco l'immagine binaria dei pixel candidati a foreground,confrontando le immagini delle inconsistenze di background
 e foreground.Inoltre effettuo le tre operazioni aggiuntive descritte nel paper per rimuovere le componenti piccole e isolate e
 affinare il contorno.
 */

void thresholdForeground(const Mat& image,const Mat& inconsistBg,const Mat& inconsistFg,Mat& classificationBN){
    const uchar *p,*q;
    classificationBN=Mat::zeros(image.rows, image.cols, CV_8U);
    for(int i=maskR.y;i<maskR.y+maskR.height;i++){
        p=inconsistBg.ptr<uchar>(i);
        q=inconsistFg.ptr<uchar>(i);
        for(int j=maskR.x;j<maskR.x+maskR.width;j++){
            if(q[j]<=p[j])
                classificationBN.at<uchar>(i,j)=255;
        }
    }
    regionSizeFiltering(classificationBN);
    referenceReachableFiltering(classificationBN);
    narrowGraphRefinement(image,classificationBN);
}












