#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include "bkgrd.h"
#include "structInconsist.h"
#include "iterAnalizer.h"

using namespace std;
using namespace cv;

Rect maskR;

int main(int argc, char** argv) {
    if(argc!=3){
        cout <<" Usage: SI-Cut ImageToLoad Mask" << endl;
        return -1;
    }
    Mat image = imread(argv[1], IMREAD_COLOR);
    Mat mask=imread(argv[2],IMREAD_GRAYSCALE);
    Mat T=mask.clone();
    Mat background(image.rows,image.cols,image.type());
    Mat V=Mat::zeros(mask.rows, mask.cols, CV_8U);
    Mat V_prec=Mat::zeros(mask.rows, mask.cols, CV_8U);
    Mat V_2prec=Mat::zeros(mask.rows, mask.cols, CV_8U);
    Mat result(image.rows,image.cols,CV_8U);
    Mat F_fb,F_bb;
    int vtcK,cur_t,t_fb,t_bb,hor_dist,ver_dist;
    float var_percent;
    char choose;
    Point tl,br;
    vector<uchar> vtcK_list; //vtcK list
    vector<Mat> T_list; //target region list
    if(!image.data || !mask.data)
    {
        cout <<  "Could not open or find the image/mask" << endl ;
        return -1;
    }
    maskR=boundingRect(mask);
    tl=maskR.tl();
    br=maskR.br();
    //Const area
    //------------------------------------------
    const int MAX_ITER_TH=10;
    const float MIN_VAR_TH=0.025; //2.5%
    const int MAX_HOR_TH=abs(tl.x-br.x)/3;
    const int MAX_VER_TH=abs(tl.y-br.y)/3;
    //------------------------------------------
    var_percent=1;
    cur_t=0;//iteration number
    
    while(1){
        cout<<"\nBACKGROUND STRUCTURAL PREDICTION"<<endl;
        backgroundSPrediction(image,T,V,background,true);
        structInconsAnalysis(image,background,T,vtcK,result);
        vtcK_list.push_back(vtcK);
        T_list.push_back(result.clone());
        cout<<"\nDo you want stop the iterations? [Y=yes,N=no]:";
        cin>>choose;
        cin.clear();
        cin.ignore(INT_MAX,'\n');
        /*
         Se l'utente preferisce l'iterazione corrente interrompo le iterazioni ed esporto il risultato per l'ottimizzazione
         del contorno
         */
        if(toupper(choose)=='Y'){
            t_fb=cur_t;
            t_bb=cur_t;
            F_fb=result.clone();
            F_bb=result.clone();
            break;
        }
        if(cur_t>0)
            regionVariationPercent(result, T_list[cur_t-1], var_percent);
        closestSideDistances(result, tl, br, hor_dist, ver_dist);
        /*
         Se si verifica uno dei tre criteri d'arresto (superato il numero massimo di iterazioni,variazione
         troppo piccola tra due iterazioni successive,bounding-box della regione T attuale troppo ridotta
         rispetto al rettangolo di input) interrompo le iterazioni
         */
        cout<<"\nITERATION:"<<cur_t<<" VARIATION:"<<var_percent*100<<"% MAX HORIZONTAL DISTANCE:"<<hor_dist<<"px MAX VERTICAL DISTANCE:"<<ver_dist<<"px"<<endl;
        if((cur_t>MAX_ITER_TH) || (var_percent<MIN_VAR_TH) || (hor_dist>MAX_HOR_TH) || (ver_dist>MAX_VER_TH)){
            bestInterval(vtcK_list, t_fb, t_bb);
            cout<<"\nt_fb:"<<t_fb<<" t_bb:"<<t_bb<<endl;
            F_fb=T_list[t_fb];
            F_bb=T_list[t_bb];
            break;
        }
        //Update V region
        V_2prec=V_prec.clone();
        V_prec=T-result;
        V=V_2prec+V_prec;
        T=result.clone();
        cur_t++;
    }
    wideBandContourOptimization(image, mask, F_fb,F_bb,result);
    return 0;
}

