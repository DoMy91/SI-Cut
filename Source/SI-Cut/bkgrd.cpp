//
//  bkgrd.cpp
//  SI-Cut
//
//  Created by Domenico on 26/11/15.
//  Copyright (c) 2015 Domenico. All rights reserved.
//
#include "bkgrd.h"
#include "GCoptimization.h"
#include <iostream>

using namespace cv;
using namespace std;

int M,N;
Rect boundRect;
int max_shift_x, min_shift_x, max_shift_y, min_shift_y;
bool isbkgrd;

struct ForDataFn{
    Mat image,mask,Vreg,gradX,gradY,shiftX,shiftY;
};

/*
 BACKGROUND STRUCTURAL PREDICTION
 */

void backgroundSPrediction(const Mat& image,const Mat& mask,const Mat& Vregion,Mat& background,bool flag){
    Mat cloneMask=mask.clone(),cloneImg=image.clone(),cloneVreg=Vregion.clone(),gradX,gradY,shiftX,shiftY;
    Mat resultImg;
    float fac=1,limit;
    const float BGD_LIMIT=1.2;
    const float FGD_LIMIT=0.5;
    int iter,nIter,cntT,num_labels;
    Vec3b *pointer;
    //Ridimensiono la maschera fino a quando il numero di pixel in T e' <=600
    while((cntT=countNonZero(cloneMask))>600){
        fac/=2;
        resize(mask,cloneMask,Size(),fac,fac,INTER_AREA);
    }
    threesholdMask(cloneMask);
    nIter=-log2(fac)+1;
    isbkgrd=flag;
    for(iter=1;iter<=nIter;iter++){
        //Parto dal livello più basso della piramide.Ad ogni iterazione effettuo l'upscaling dell' immagine
        resize(image, cloneImg, Size(),fac,fac,INTER_AREA);
        resize(Vregion, cloneVreg,Size(),fac,fac,INTER_AREA);
        threesholdMask(cloneVreg);
        computeGradient(cloneImg,gradX,gradY);
        M=cloneImg.cols;
        N=cloneImg.rows;
        cout<<"\niteration n."<<iter<<" img dimension:["<<N<<"x"<<M<<"]"<<endl;
        if(iter==1){
            /*
             Invece di considerare tutti i possibili shift [ x=>-(M-1),M-1 , y=> -(N-1),N-1]
             calcolo il bounding-rect della regione T attuale e considero shift massimi pari alle
             dimensioni di tale rettangolo incrementate del 20% (nel caso di stima di background).
             In tal modo riduco il numero di shift (e quindi di label) da considerare.
             */
            boundRect=boundingRect(cloneMask);
            if(isbkgrd)
                limit=BGD_LIMIT;
            else
                limit=FGD_LIMIT;
            min_shift_x=-(boundRect.width*limit-1);
            max_shift_x=boundRect.width*limit-1;
            min_shift_y=-(boundRect.height*limit-1);
            max_shift_y=boundRect.height*limit-1;
            shiftX=Mat::zeros(N,M,CV_16S);
            shiftY=Mat::zeros(N,M,CV_16S);
            num_labels=(max_shift_x-min_shift_x+1)*(max_shift_y-min_shift_y+1);
            gridGraphIndividually(num_labels,cloneImg, cloneMask,cloneVreg-cloneMask, gradX, gradY, shiftX, shiftY);
        }
        else{
            //ridimensiono la maschera per farne combaciare la dimensione con l'immagine corrente
            resize(mask, cloneMask, Size(),fac,fac,INTER_AREA);
            threesholdMask(cloneMask);
            boundRect=boundingRect(cloneMask);
            //propago i vettori di shift al livello successivo
            interpNearestShift(shiftX, shiftY, cloneMask);
            min_shift_x=-1;
            max_shift_x=1;
            min_shift_y=-1;
            max_shift_y=1;
            num_labels=18; //(9 move+9 neighbor leap)
            gridGraphIndividually(num_labels,cloneImg, cloneMask, cloneVreg-cloneMask, gradX, gradY, shiftX, shiftY);
        }
        fac*=2;
    }
    cout<<"\nDONE"<<endl;
    //Visualizzo l'immagine corrispondente al labeling
    for(int i=0;i<N;i++){
        pointer=cloneImg.ptr<Vec3b>(i);
        for(int j=0;j<M;j++){
            pointer[j]=cloneImg.at<Vec3b>(i+shiftY.at<short>(i,j),j+shiftX.at<short>(i, j));
        }
    }
    imshow("Background",cloneImg);
    waitKey();
    background=cloneImg;
}

void gridGraphIndividually(int num_labels,const Mat& image,const Mat& mask,const Mat& Vreg,const Mat& gradX,const Mat& gradY,Mat& shiftX,Mat& shiftY){
    float oldEnergy,newEnergy=0;
    try{
        GCoptimizationGridGraph *gc=new GCoptimizationGridGraph(boundRect.width,boundRect.height,num_labels);
        ForDataFn fn;
        fn.image=image;
        fn.mask=mask;
        fn.Vreg=Vreg;
        fn.gradX=gradX;
        fn.gradY=gradY;
        fn.shiftX=shiftX;
        fn.shiftY=shiftY;
        gc->setDataCost(&dataFn,&fn);
        gc->setSmoothCost(&SmoothCostFn,&fn);
        int n_iter=0;
        do{
            oldEnergy=newEnergy;
            gc->expansion(1);
            newEnergy=gc->compute_energy();
            cout<<".";
            n_iter++;
        }while(abs(oldEnergy-newEnergy)>5 && n_iter<20);
        Mat tmpShiftX,tmpShiftY;
        tmpShiftX=shiftX.clone();
        tmpShiftY=shiftY.clone();
        /*
         Costruisco le matrici degli shift a partire dal labeling prodotto dall'algoritmo di a-expansion
         e dalle stesse matrici del passo precedente
         */
        for(int i=0;i<boundRect.width*boundRect.height;i++){
            int px,py,dx,dy,lab;
            lab=gc->whatLabel(i);
            ind2sub(i, px, py);
            //cout<<"x:"<<px<<" y:"<<py<<" label:"<<lab<<endl;
            if(lab<(max_shift_x-min_shift_x+1)*(max_shift_y-min_shift_y+1)){ //move label
                lab2shift(lab, dx, dy);
                tmpShiftX.at<short>(py,px)=shiftX.at<short>(py,px)+dx;
                tmpShiftY.at<short>(py,px)=shiftY.at<short>(py,px)+dy;
            }
            else{ //leap label
                lab%=(max_shift_x-min_shift_x+1)*(max_shift_y-min_shift_y+1);
                lab2shift(lab, dx, dy);
                tmpShiftX.at<short>(py,px)=shiftX.at<short>(py+dy,px+dx)+dx;
                tmpShiftY.at<short>(py,px)=shiftY.at<short>(py+dy,px+dx)+dy;
            }
        }
        shiftX=tmpShiftX;
        shiftY=tmpShiftY;
    }
    catch(GCException e){
        e.Report();
    }
}

float dataFn(int site,int label,void *d){
    ForDataFn *data=(ForDataFn *)d;
    int p_x,p_y,sp_x,sp_y,i,j,u_x,u_y,u2_x,u2_y;
    Vec3b i_u,i_u2;
    short gx_u,gy_u,gx_u2,gy_u2;
    float energy=0,w_brd=0.5;
    ind2sub(site, p_x, p_y);
    if(!computeShift(p_x, p_y, label, data->shiftX, data->shiftY, data->mask, sp_x, sp_y))
        return GCO_MAX_ENERGYTERM;
    //lo shift non può portare il punto p fuori dall'immagine
    if(!isInside(p_x+sp_x,p_y+sp_y))
        return GCO_MAX_ENERGYTERM;
    //i punti esterni alla regione T devono avere shift nullo
    if((int)(data->mask.at<uchar>(p_y,p_x)) == 0 && (sp_x!=0 || sp_y!=0))
        return GCO_MAX_ENERGYTERM;
    //i punti interni alla regione T devono essere collegati con pixel appartenenti alla regione R
    if((int)(data->mask.at<uchar>(p_y+sp_y,p_x+sp_x)) ==1)
        return GCO_MAX_ENERGYTERM;
    //BORDER CONSISTENCY
    //se il punto appartiene alla regione T analizzo un suo intorno 3x3
    if((int)(data->mask.at<uchar>(p_y,p_x)) == 1){
        //non posso collegare il punto della regione T ad un punto della regione V
        if((int)(data->Vreg.at<uchar>(p_y+sp_y,p_x+sp_x)) ==1)
            return GCO_MAX_ENERGYTERM;
        for(i=-1;i<=1;i++){
            for(j=-1;j<=1;j++){
                if(i!=0 || j!=0){
                    u_x=p_x+j;
                    u_y=p_y+i;
                    //vicino di P appartenente alla regione R U V [punto u]
                    if(isInside(u_x,u_y) && (int)(data->mask.at<uchar>(u_y,u_x)) == 0){
                        i_u=data->image.at<Vec3b>(u_y,u_x);
                        gx_u=data->gradX.at<short>(u_y,u_x);
                        gy_u=data->gradY.at<short>(u_y,u_x);
                        u2_x=p_x+sp_x+j;
                        u2_y=p_y+sp_y+i;
                        //il punto u' deve appartenere alla regione R U V
                        if(!isInside(u2_x, u2_y) || (int)(data->mask.at<uchar>(u2_y,u2_x))==1)
                            return GCO_MAX_ENERGYTERM;
                        else{
                            i_u2=data->image.at<Vec3b>(u2_y,u2_x);
                            gx_u2=data->gradX.at<short>(u2_y,u2_x);
                            gy_u2=data->gradY.at<short>(u2_y,u2_x);
                            energy+=w_brd*(norm((Scalar)i_u-(Scalar)i_u2)+(abs(gx_u-gx_u2)+abs(gy_u-gy_u2)));
                        }
                    }
                }
            }
        }
        if(!isbkgrd) //only reverse structural prediction
            energy+=LocationPenalty(Point(p_x,p_y), Point(p_x+sp_x,p_y+sp_y), data->image.cols, data->image.rows);
    }
    return energy;
}

float SmoothCostFn(int s1,int s2,int l1,int l2,void *d){
    ForDataFn *data=(ForDataFn *)d;
    int p_x,p_y,sp_x,sp_y,q_x,q_y,sq_x,sq_y;
    Vec3b i_rp,i_rq,i_rpN,i_rqN;
    short gx_rp,gy_rp,gx_rq,gy_rq,gx_rpN,gy_rpN,gx_rqN,gy_rqN;
    float energy=0,w_nb=0.1875;
    ind2sub(s1, p_x, p_y);
    ind2sub(s2, q_x, q_y);
    if(!computeShift(p_x, p_y, l1, data->shiftX, data->shiftY, data->mask, sp_x, sp_y))
        return GCO_MAX_ENERGYTERM;
    if(!computeShift(q_x, q_y, l2, data->shiftX, data->shiftY, data->mask, sq_x, sq_y))
        return GCO_MAX_ENERGYTERM;
    //due pixel adiacenti con stesso valore di shift hanno un costo di smoothness nullo
    if(sp_x==sq_x && sp_y==sq_y)
        return 0;
    //i punti R(P),R(Q),R(P)+(Q-P) e R(Q)+(P-Q) devono appartenere all'immagine
    if(!isInside(p_x+sp_x, p_y+sp_y) || !isInside(q_x+sq_x, q_y+sq_y))
        return GCO_MAX_ENERGYTERM;
    if(!isInside(sp_x+q_x,sp_y+q_y) || !isInside(sq_x+p_x, sq_y+p_y))
        return GCO_MAX_ENERGYTERM;
    
    i_rp=data->image.at<Vec3b>(p_y+sp_y,p_x+sp_x);
    gx_rp=data->gradX.at<short>(p_y+sp_y,p_x+sp_x);
    gy_rp=data->gradY.at<short>(p_y+sp_y,p_x+sp_x);
    
    i_rpN=data->image.at<Vec3b>(sp_y+q_y,sp_x+q_x);
    gx_rpN=data->gradX.at<short>(sp_y+q_y,sp_x+q_x);
    gy_rpN=data->gradY.at<short>(sp_y+q_y,sp_x+q_x);
    
    i_rq=data->image.at<Vec3b>(q_y+sq_y,q_x+sq_x);
    gx_rq=data->gradX.at<short>(q_y+sq_y,q_x+sq_x);
    gy_rq=data->gradY.at<short>(q_y+sq_y,q_x+sq_x);
    
    i_rqN=data->image.at<Vec3b>(sq_y+p_y,sq_x+p_x);
    gx_rqN=data->gradX.at<short>(sq_y+p_y,sq_x+p_x);
    gy_rqN=data->gradY.at<short>(sq_y+p_y,sq_x+p_x);
    
    energy+=norm((Scalar)i_rp-(Scalar)i_rqN)+norm((Scalar)i_rq-(Scalar)i_rpN)+(abs(gx_rp-gx_rqN)+abs(gy_rp-gy_rqN)+abs(gx_rq-gx_rpN)+abs(gy_rq-gy_rpN));
    
    return energy*w_nb;
}

float LocationPenalty(Point p,Point r_P,float width,float height){
    Point s_P=r_P-p;
    float xMax,xFree,yMax,yFree,cxMax,cyMax,energy=0,w_loc=1;
    xFree=width*25/100;
    xMax=width*50/100;
    yFree=height*15/100;
    yMax=height*45/100;
    cxMax=100;//??
    cyMax=100;//??
    if(abs(s_P.x)>=xMax)
        energy+=cxMax;
    else if(!(abs(s_P.x)<=xFree))
        energy+=(cxMax*(abs(s_P.x)-xFree))/(xMax-xFree);
    if(abs(s_P.y)>=yMax)
        energy+=cyMax;
    else if(!(abs(s_P.y)<=yFree))
        energy+=(cyMax*(abs(s_P.y)-yFree))/(yMax-yFree);
    return energy*w_loc;
}

/*
 A partire dalle coordinate di un pixel,dalla label e dalle matrici di shift attuali calcolo le componenti x e y
 del vettore di shift s(p)
 */
bool computeShift(int p_x,int p_y,int label,const Mat& shiftX,const Mat& shiftY,const Mat& mask,int& sp_x,int& sp_y){
    if(label<(max_shift_x-min_shift_x+1)*(max_shift_y-min_shift_y+1)){ //move label
        lab2shift(label, sp_x, sp_y);
        sp_x+=shiftX.at<short>(p_y,p_x);
        sp_y+=shiftY.at<short>(p_y,p_x);
    }
    else { //leap label
        label%=(max_shift_x-min_shift_x+1)*(max_shift_y-min_shift_y+1);
        lab2shift(label, sp_x, sp_y);
        /*
         sp_x=0 && sp_y=0 si verifica solo se la label è 13.Non sono interessato a tale label in
         quanto è equivalente alla move label 4.
         Procedo se il punto vicino in esame (p_x+sp_x,p_y+sp_y) appartiene alla regione T
         */
        if((sp_x==0 && sp_y==0) || !isInside(p_x+sp_x, p_y+sp_y)  || (int)(mask.at<uchar>(p_y+sp_y,p_x+sp_x)) == 0)
            return false;
        sp_x+=shiftX.at<short>(p_y+sp_y,p_x+sp_x);
        sp_y+=shiftY.at<short>(p_y+sp_y,p_x+sp_x);
    }
    return true;
}


//
// Convert between 1D indices and 2D
//
void ind2sub(int ind, int& x, int& y)
{
    x = (ind%boundRect.width)+boundRect.tl().x;
    y = (ind/boundRect.width)+boundRect.tl().y;
}

//
// Convert between labels and actual shifts
//
int shift2lab(int dx, int dy)
{
    return (dx - min_shift_x) + (max_shift_x-min_shift_x+1)*(dy - min_shift_y);
}

void lab2shift(int lab, int& dx, int& dy)
{
    dx = lab % (max_shift_x-min_shift_x+1);
    dy = (lab-dx)/(max_shift_x-min_shift_x+1) + min_shift_y;
    dx = dx  + min_shift_x;
}

//calcolo del gradiente in direzione x e y con maschere di Sobel
void computeGradient(const Mat& image,Mat& gradX,Mat& gradY){
    Mat gradient;
    cvtColor(image,gradient,COLOR_RGB2GRAY);
    /*
     scale=0.125 in quanto voglio scalare i risultati nell'intervallo [-128,127].
     Mediante maschera di Sobel il valore massimo del gradiente in un punto è pari a 1020 (255*4).
     y:1020=x:127  ==>x=y*(127/1020)=y*0.125
     */
    Sobel(gradient,gradX,CV_16S,1,0,3,0.125,0,BORDER_DEFAULT);
    Sobel(gradient,gradY,CV_16S,0,1,3,0.125,0,BORDER_DEFAULT);
}

//Controllo se il punto P(p_x,p_y) appartiene all'immagine
bool isInside(int p_x,int p_y){
    return !(p_x<0 || p_x>=M || p_y<0 || p_y>=N);
}

//Tale function setta a 1 tutti i pixel della maschera che sono diversi da 0.L'insieme di tutti questi pixel costituisce la regione T
void threesholdMask(Mat& mask){
    int i,j;
    uchar *p;
    for(i=0;i<mask.rows;i++){
        p=mask.ptr<uchar>(i);
        for(j=0;j<mask.cols;j++){
            if(p[j]!=0)
                p[j]=1;
        }
    }
}

/*
 Mediante tale function propago i vettori di shift al livello successivo,raddoppiandone i valori e la dimensione delle matrici,utilizzando l'interpolazione al più vicino (nearest neighbor interpolation)
 */
void interpNearestShift(Mat& shiftX,Mat& shiftY,Mat& mask){
    int i,j;
    short *p,*q;
    uchar *r;
    //Raddoppio i valori degli shift
    for(i=0;i<shiftX.rows;i++){
        p=shiftX.ptr<short>(i);
        q=shiftY.ptr<short>(i);
        for(j=0;j<shiftX.cols;j++){
            p[j]*=2;
            q[j]*=2;
        }
    }
    //Raddoppio la dimensione delle matrici contenenti gli shift e interpolo
    resize(shiftX, shiftX, mask.size(),2,2,INTER_NEAREST);
    resize(shiftY,shiftY,mask.size(),2,2,INTER_NEAREST);
    //Setto shift nulli per tutti i pixel che non fanno parte della regione T
    for(i=0;i<shiftX.rows;i++){
        p=shiftX.ptr<short>(i);
        q=shiftY.ptr<short>(i);
        r=mask.ptr<uchar>(i);
        for(j=0;j<shiftX.cols;j++){
            if(r[j]==0){
                p[j]=0;
                q[j]=0;
            }
        }
    }
}



