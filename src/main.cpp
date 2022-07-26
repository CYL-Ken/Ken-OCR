//============================================================================
// Name        : OCR
// Author      : Ken
// Version     : 1.9.6
// Copyright   : CYL-TEK reserved
// Description : OCR_DEMO 
//============================================================================

#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <chrono>
#include <future>
#include <ctime>
#include <algorithm>
#include <unistd.h>
//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//中文字轉換
#include "CvxText.h"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

//長庚資料傳送
#include "CgmhData.hpp"

//log
#include "loguru.hpp"

//Tensorflowlite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"

#define WINDOW_NAME "Cyl-Tek Thermal"
#define WINDOW1_NAME "Cyl-Tek ID Box"

using namespace cv;
using namespace std;


// data structure for storing the inference result
typedef struct
{
    int label_index;
    float prob;
} info_t;


void onMouse(int Event,int x,int y,int flags,void* param);

void onMouse(int Event,int x,int y,int flags,void* param){
    if(Event==EVENT_LBUTTONDOWN){
        cout << "X = " << x<<"Y = " << y << endl;
       
    }
   
}



//系統參數
typedef struct SysSet {
    
    
    string file="OCR.txt";
    string sshPath ="| ssh root@192.168.53.32 -T \" cat > /data/OCR.txt\"";
    string RTSPpath="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov";
    string RFIDPath="/home/pi/RFID/uid.txt";
    string SmartCardPath="/home/pi/RFID/uid.txt";
    string SCPRath="";
    string RTSP_Img="";
    string CgmhUrl="";
    int cap_width=1280;
    int cap_height=1024;
    int Card_PixX=800;
    int Card_PixY=500;
    int PassPort_PixX = 1650;
    
    int Fov_X=0;
    int Fov_Y=0;
    int Fov_W=1280;
    int Fov_H=730;

    int H_min=0;
    int H_max=180;
    int S_min=0;
    int S_max=255;
    int V_min=200;
    int V_max=255;
    
    int Card_Type_num=1;

    //const char* OCR_Chi="chi_O"; 
    //const char* OCR_Chi="chi_O";
    const char* OCR_Eng="eng";
    //const char* OCR_Chi_ID="chi_tra1603S";
    
    //const char* OCR_Chi="tch_full_detect_1006_88x88.tflite";
    //const char* OCR_Chi="tch_detect_0913_88x88.tflite";
    //const char* OCR_Chi="tch_detect_236_1012.tflite";
    const char* OCR_Chi="tch_detect_1685_88x88_1021.tflite";
    
    //const char* OCR_Chi="tch_detect.tflite";
    //const char* OCR_Chi_ID="chi_traFS";
    
    //const char* _tflife_path="Card_cal_detect.tflite";
    //const char* _tflife_path="Card_cal_detect_0913_96x96.tflite";
    const char* _tflife_path="Card_cal_detect_0924_128X128.tflite";
   
    
    bool show_r=true;
          
    bool bSeting = false;
    bool bSave=false;
    bool bLog=false;
    bool bSave_Img=false;
    bool send_commd=false;
    bool bUse_RFID=false;
    bool bUse_SmartCard=false;
    bool bUse_PassPort = false;
    bool offline=false;
    string off_ImgePath="";
    int OCRTimeFind=200;
    int OCRTimeCunt=0;

    
} SysSet_t;

//卡片設定參數
typedef struct  CardSet {
    
    bool bID=true;
    bool bName=true;
    float TempScore=0.75;
     
    int H_min=0;
    int H_max=180;
    int S_min=0;
    int S_max=255;
    int V_min=200;
    int V_max=255;
    
    int lowThreshold = 50;
    int ratio = 3;
    int kernel_size = 3;
    
    int OCR_Processing_type=0; //0:HSV ,1:B_Channel,2:G_Channel,3:R_Channel
    string Card_name;
    bool bMrop_Erode_ID=true;
    bool bMrop_Dilate_ID=true;
    int ID_th_low=135;
    int ID_th_high=255;
    int Erode_ID_size=3;
    int Dilate_ID_size=3;
    int ID_top_X=250;
    int ID_top_Y=280;
    int ID_Down_X=500;
    int ID_Down_Y=380;
    int ID_H_th=255;
    int ID_L_th=95;
    int ID_Roi_width=130;
    int ID_Roi_high=75;
    
    
    bool bMrop_Erode_Name=true;
    bool bMrop_Dilate_Name=true;
    int Erode_Name_size=3;
    int Dilate_Name_size=3;
    int Name_th_low=135;
    int Name_th_high=255;
    int Name_top_X=260;
    int Name_top_Y=170;
    int Name_Down_X=520;
    int Name_Down_Y=320;
    int Name_H_th=255;
    int Name_L_th=200;
    int Name_Roi_width=110;
    int Name_Roi_high=110;
    
    
} CardSet_t;

//TF model Set
typedef struct  modelSet {
    
    int model_width;
    int model_height;
    int model_channels;
    int In;
    int Label_dict;
    
} modelSet_t;

//OCR result
typedef struct  OCRresult {
    
   string Name;
   string ID;
   string Data;
   Mat Card_Img;
    
} OCRresult_t;

//RFID Set
typedef struct  RFIDSet {
    
    string RFID_UID;
    string Name;
    string ID;
      
} RFIDSet_t;

//護照設定參數
typedef struct  PassPortSet {

    bool bID = true;
    bool bName = true;
 
    int H_min = 0;
    int H_max = 180;
    int S_min = 0;
    int S_max = 255;
    int V_min = 200;
    int V_max = 255;
   

    int lowThreshold = 50;
    int max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;

    int OCR_Processing_type = 0; //0:HSV ,1:B_Channel,2:G_Channel,3:R_Channel
    string Card_name;
    bool bMrop_Erode_ID = true;
    bool bMrop_Dilate_ID = true;
    int ID_th_low = 135;
    int ID_th_high = 255;
    int Erode_ID_size = 3;
    int Dilate_ID_size = 3;
    int ID_top_X = 250;
    int ID_top_Y = 280;
    int ID_Down_X = 500;
    int ID_Down_Y = 380;
    int ID_H_th = 255;
    int ID_L_th = 95;
    int ID_Roi_width = 130;
    int ID_Roi_high = 75;


    bool bMrop_Erode_Name = true;
    bool bMrop_Dilate_Name = true;
    int Erode_Name_size = 3;
    int Dilate_Name_size = 3;
    int Name_th_low = 135;
    int Name_th_high = 255;
    int Name_top_X = 250;
    int Name_top_Y = 180;
    int Name_Down_X = 520;
    int Name_Down_Y = 320;
    int Name_H_th = 255;
    int Name_L_th = 200;
    int Name_Roi_width = 110;
    int Name_Roi_high = 110;


} PassPortSet_t;

//OCR 引擎物件
tesseract::TessBaseAPI *api_eng_id;

//Tensorflowlite  卡片識別物件
unique_ptr<tflite::Interpreter> interpreter;
unique_ptr<tflite::FlatBufferModel> model;
tflite::ops::builtin::BuiltinOpResolver resolver;

//Tensorflowlite OCR物件
unique_ptr<tflite::Interpreter> interpreter_OCR;
unique_ptr<tflite::FlatBufferModel> model_OCR;
tflite::ops::builtin::BuiltinOpResolver resolver_OCR;
vector<string> Labels;

modelSet_t Model_Set; //TF 卡片識別模型參數
modelSet_t OcRModel_Set; //TF OCR模型參數
info_t *data = (info_t *)malloc(1685 * sizeof(info_t));

Mat Nameimg;      //名子圖片
Mat IDimg;        //ID圖片

Mat frame;        //當次相機影像
Mat r_frame;      //結果圖片
Mat FoV_img;      //FoV影像
Mat hsv,r_hsv;    //轉到hsv平面
Mat dst;          //備份原圖影像

Mat RFID_img=Mat(250, 768, CV_8UC3);

Mat I_rgb_warped;  //RGB卡片影像
Mat I_hsv_warped;  //HSV卡片影像

// warp card
Point2f sp[4];
double d1,d2;
RotatedRect rRect;

SysSet_t Seting;   //系統設定
CardSet_t Card[2]; //卡片設定
vector<OCRresult_t> _OCRresult; //OCR結果
vector<RFIDSet_t> _RFIDdata;

string CURR_RFID = "START";
string PREV_RFID;

//Set file Path
string OCRfile_Name="OCR_file.xml";


bool bOCR=true;            //OCR開始旗標
bool bCardDetect=false;    //是否檢測Card旗標
bool bWarn=false;

bool bOCR_Start=false;     //OCR Start
bool bOCROK=false;         //OCR OK
bool bOCRNG=false;         //OCR NG
bool bOCR_End=true;        //OCR End

bool bRFIDOK=false;        //RFID OK
bool bRFIDNG=false;        //RFID NG
bool RFID_send=false;      //RFID 結果傳送
int rRFID=-1;              //Smart卡片Type

int Type_num=-1;           //卡片Type num ex:  -1初始直 0健保卡 1身分證

string Text_num;           //ID文字
string Text_name;          //名子文字
string rText_num;
string rText_name;

string rfid_Text_num;           //ID文字
string rfid_Text_name; 
string rfid_text_id;
string rfid_text_name;
wchar_t *w_ID;
wchar_t *w_str;

int L_cunt=0;              //顯示計數

//TimeOut
auto TimeOutStart=std::chrono::steady_clock::now();;
auto TimeOutEnd=std::chrono::steady_clock::now();; 

//座標排需左上、右上、左下、右下
struct str{
    bool operator() ( Point2f a, Point2f b ){
        if ( a.y != b.y ) 
            return a.y < b.y;
        return a.x <= b.x ;
    }
} comp;


// comparison function object
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

// compare function for qsort
int compare(void const *a, void const *b)
{
    info_t *x, *y;

    x = (info_t *)a;
    y = (info_t *)b;

    return (x->prob < y->prob)? 1: -1;
}

//取得系統當下時間並轉成字串
string getCurrentSystemTime() 
{
    auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* ptm = localtime(&tt);
    char date[60] = { 0 };
    sprintf(date, "%d-%02d-%02d-%02d.%02d.%02d",
        (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
        (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
        
    return std::string(date);
}

//Char to Wchar
static int ToWchar(char* &src, wchar_t* &dest, const char *locale = "zh_CN.utf8")
{
    //cout << "ToWchar Start"  << std::endl;
    if (src == NULL) {
        dest = NULL;
        return 0;
    }

    // 根據環境變量設置locale
    setlocale(LC_CTYPE, locale);

    // 得到轉化爲需要的寬字符大小
    int w_size = mbstowcs(NULL, src, 0) + 1;

    // w_size = 0 說明mbstowcs返回值爲-1。即在運行過程中遇到了非法字符(很有可能使locale
    // 沒有設置正確)
    if (w_size == 0) {
        dest = NULL;
        return -1;
    }

    //wcout << "w_size" << w_size << endl;
    dest = new wchar_t[w_size];
    if (!dest) {
        return -1;
    }

    int ret = mbstowcs(dest, src, strlen(src)+1);
    if (ret <= 0) {
        return -1;
    }
    //cout << "ToWchar End" << std::endl;
    return 0;
   
}

//初始化系統
int initTesseract(SysSet_t Set){
    api_eng_id = new tesseract::TessBaseAPI();
    if (api_eng_id->Init(NULL, Set.OCR_Eng)) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);}
    api_eng_id->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    api_eng_id->SetVariable("tessedit_char_blacklist", " .|*%$#@!*(){}[]§");   
    cout << "[System State]Initial Tesseract Complete" << endl;
    
    return 0;
}

//初始化OCR*
int initSystem(string Set_xml, SysSet_t &Set, CardSet_t Card_set[]){
    
    FileStorage fs(Set_xml, FileStorage::READ);
    
    fs["Card_Type_num"] >> Set.Card_Type_num;
    fs["Cap_Widrh"] >> Set.cap_width;
    fs["Cap_Height"] >> Set.cap_height;
    fs["Cap_Fov_X"] >> Set.Fov_X;
    fs["Cap_Fov_Y"] >> Set.Fov_Y;
    fs["Cap_Fov_W"] >> Set.Fov_W;
    fs["Cap_Fov_H"] >> Set.Fov_H;
    fs["Base_Card_PixX"] >> Set.Card_PixX;
    fs["Base_Card_PixY"] >> Set.Card_PixY;
    fs["Base_H_min"] >> Set.H_min;
    fs["Base_H_max"] >> Set.H_max;
    fs["Base_S_min"] >> Set.S_min;  
    fs["Base_S_max"] >> Set.S_max;
    fs["Base_V_min"] >> Set.V_min;
    fs["Base_V_max"] >> Set.V_max; 
    fs["Use_Cgmhsend"] >> Set.send_commd;
    fs["Save_OcrFile_Name"] >> Set.file;
    fs["SSH_Path"] >> Set.sshPath;
    fs["RTSP_path"] >> Set.RTSPpath;
    fs["SmartCardPath"]>>Set.SmartCardPath;
    fs["RFIDPath"] >> Set.RFIDPath;
    fs["SCPRath"] >> Set.SCPRath;
    fs["RTSP_Img"] >> Set.RTSP_Img;
    
    fs["CgmhUrl"] >> Set.CgmhUrl;
    
    fs["Show_Win"] >> Set.show_r; 
    fs["offline"] >> Set.offline;
    fs["off_ImgePath"] >> Set.off_ImgePath;
    
    
    //fs["OCR_Chi"] >> Set.OCR_Chi;
    //fs["OCR_Eng"] >> Set.OCR_Eng;
    fs["bLog"] >> Set.bLog; 
    fs["bSave_Img"] >> Set.bSave_Img;
    fs["bUse_RFID"] >> Set.bUse_RFID;
    fs["bUse_SmartCard"] >> Set.bUse_SmartCard;
    fs["bUse_PassPort"] >> Set.bUse_PassPort;
  
    Card_set[0].Card_name="健保卡";
    fs["TempScore1"]>>Card_set[0].TempScore;
    fs["OCR_Processing_type1"]>>Card_set[0].OCR_Processing_type;
    fs["bID1"] >> Card_set[0].bID;
    fs["bName1"] >> Card_set[0].bName;
    fs["H_min1"] >> Card_set[0].H_min;  
    fs["H_max1"] >> Card_set[0].H_max;
    fs["S_min1"] >> Card_set[0].S_min;
    fs["S_max1"] >> Card_set[0].S_max; 
    fs["V_min1"] >> Card_set[0].V_min;
    fs["V_max1"] >> Card_set[0].V_max;
    fs["lowThreshold1"]>>Card_set[0].lowThreshold;
    fs["ratio1"]>>Card_set[0].ratio;
    fs["kernel_size1"]>>Card_set[0].kernel_size;
    
        
    fs["bMrop_Erode_ID1"] >> Card_set[0].bMrop_Erode_ID;
    fs["bMrop_Dilate_ID1"] >> Card_set[0].bMrop_Dilate_ID;  
    fs["Erode_ID_size1"] >> Card_set[0].Erode_ID_size;
    fs["Dilate_ID_size1"] >> Card_set[0].Dilate_ID_size;
   
    fs["ID_top_X1"] >>  Card_set[0].ID_top_X;
    fs["ID_top_Y1"] >> Card_set[0].ID_top_Y; 
    fs["ID_Down_X1"] >> Card_set[0].ID_Down_X;
    fs["ID_Down_Y1"] >> Card_set[0].ID_Down_Y;   
    fs["ID_Roi_width1"]>>Card_set[0].ID_Roi_width;
    fs["ID_Roi_high1"]>>Card_set[0].ID_Roi_high;
 
        
    fs["bMrop_Erode_Name1"] >> Card_set[0].bMrop_Erode_Name;
    fs["bMrop_Dilate_Name1"] >> Card_set[0].bMrop_Dilate_Name;  
    fs["Erode_Name_size1"] >> Card_set[0].Erode_Name_size;
    fs["Dilate_Name_size1"] >> Card_set[0].Dilate_Name_size; 
    fs["Name_th_low1"]>>Card_set[0].Name_th_low;
    fs["Name_th_high1"]>>Card_set[0].Name_th_high;
    fs["Name_top_X1"] >>  Card_set[0].Name_top_X;
    fs["Name_top_Y1"] >> Card_set[0].Name_top_Y; 
    fs["Name_Down_X1"] >> Card_set[0].Name_Down_X;
    fs["Name_Down_Y1"] >> Card_set[0].Name_Down_Y;
    fs["Name_Roi_width1"]>>Card_set[0].Name_Roi_width;
    fs["Name_Roi_high1"]>>Card_set[0].Name_Roi_high;
    
    Card_set[1].Card_name="身分證";
    fs["TempScore2"]>>Card_set[1].TempScore;
    fs["OCR_Processing_type2"]>>Card_set[1].OCR_Processing_type;
    fs["bID2"] >> Card_set[1].bID;
    fs["bName2"] >> Card_set[1].bName;
    fs["H_min2"] >> Card_set[1].H_min;  
    fs["H_max2"] >> Card_set[1].H_max;
    fs["S_min2"] >> Card_set[1].S_min;
    fs["S_max2"] >> Card_set[1].S_max; 
    fs["V_min2"] >> Card_set[1].V_min;
    fs["V_max2"] >> Card_set[1].V_max;
    fs["lowThreshold2"]>>Card_set[1].lowThreshold;
    fs["ratio2"]>>Card_set[1].ratio;
    fs["kernel_size2"]>>Card_set[1].kernel_size;
        
    fs["bMrop_Erode_ID2"] >> Card_set[1].bMrop_Erode_ID;
    fs["bMrop_Dilate_ID2"] >> Card_set[1].bMrop_Dilate_ID;  
    fs["Erode_ID_size2"] >> Card_set[1].Erode_ID_size;
    fs["Dilate_ID_size2"] >> Card_set[1].Dilate_ID_size;
    fs["ID_th_low2"]>>Card_set[1].ID_th_low;
    fs["ID_th_high2"]>>Card_set[1].ID_th_high;
    fs["ID_top_X2"] >>  Card_set[1].ID_top_X;
    fs["ID_top_Y2"] >> Card_set[1].ID_top_Y; 
    fs["ID_Down_X2"] >> Card_set[1].ID_Down_X;
    fs["ID_Down_Y2"] >> Card_set[1].ID_Down_Y; 
    fs["ID_Roi_width2"]>>Card_set[1].ID_Roi_width;
    fs["ID_Roi_high2"]>>Card_set[1].ID_Roi_high;  
        
    fs["bMrop_Erode_Name2"] >> Card_set[1].bMrop_Erode_Name;
    fs["bMrop_Dilate_Name2"] >> Card_set[1].bMrop_Dilate_Name;  
    fs["Erode_Name_size2"] >> Card_set[1].Erode_Name_size;
    fs["Dilate_Name_size2"] >> Card_set[1].Dilate_Name_size; 
    fs["Name_th_low2"]>>Card_set[1].Name_th_low;
    fs["Name_th_high2"]>>Card_set[1].Name_th_high;
    fs["Name_top_X2"] >>  Card_set[1].Name_top_X;
    fs["Name_top_Y2"] >> Card_set[1].Name_top_Y; 
    fs["Name_Down_X2"] >> Card_set[1].Name_Down_X;
    fs["Name_Down_Y2"] >> Card_set[1].Name_Down_Y;
    fs["Name_Roi_width2"]>>Card_set[1].Name_Roi_width;
    fs["Name_Roi_high2"]>>Card_set[1].Name_Roi_high;      
      
   
    fs.release();
    cout << "[System State]Initial System Setting complete" << endl;   
    
    return 0;
}

int getRFIDdata(string FilePath)
{

    
    string delimiter = ",";
    ifstream in(FilePath);
    if(!in.is_open()) return -1;
   
    
    string str;
    int num;  
    RFIDSet_t t;
    while(getline(in, str))
    {
        int s_position = str.find_first_of(delimiter);
        int e_position = str.find_last_of(delimiter);
        //cout <<  s_position<< endl;
        //cout <<  e_position << endl;
        //waitKey(0);
        t.RFID_UID=str.substr(0, s_position);
        t.Name=str.substr(s_position + 1 ,e_position-s_position-1); 
        t.ID=str.substr(e_position+ 1 , -1); 
        //cout <<  t.RFID_UID << endl;
        //cout <<  t.Name << endl;
        //cout << t.ID << endl;
         
        if(str.size()>0) 
        
        _RFIDdata.push_back(t);
    }
    in.close();
    //waitKey(0);
    return 0;
}

//TFLife init model
int init_TFLife_model(const char* tflife_path,modelSet_t &Set,
unique_ptr<tflite::Interpreter> &interpreter,unique_ptr<tflite::FlatBufferModel> &model,
tflite::ops::builtin::BuiltinOpResolver &resolver) 
{
    model = tflite::FlatBufferModel::BuildFromFile(tflife_path);
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        cout << "InterpreterBuilder failed";
    }
    interpreter->AllocateTensors();
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      //quad core

    // Get input dimension from the input tensor metadata
    // Assuming one input only
    Set.In = interpreter->inputs()[0];
    Set.model_height   = interpreter->tensor(Set.In)->dims->data[3];
    Set.model_width    = interpreter->tensor(Set.In)->dims->data[2];
    Set.model_channels = interpreter->tensor(Set.In)->dims->data[1];

    cout << "Model Information:" << endl;    
    cout << "In   : "<< Set.In << endl;
    cout << "height   : "<< Set.model_height << endl;
    cout << "width    : "<< Set.model_width << endl;
    cout << "channels : "<< Set.model_channels << endl;

  	return 0;
}

//GetImageTFLite
void GetImageTFLite(float* out, Mat src,modelSet_t Set)
{
    //imshow( "Card_Img", src );
    //waitKey(0);
    int i,Len;
    int c=0;
    float f;
    uint8_t *in;
    static Mat image;
    Mat rgbchannel[3];
    // copy image to input as input tensor
    cv::resize(src, image, Size(Set.model_width,Set.model_height),INTER_AREA);
   
    if(image.channels()<3)
    {
        cvtColor(image, image, COLOR_GRAY2RGB);
        //cout<<"image.channels() :"<<image.channels()<<endl;
    } 
    
    split(image, rgbchannel);
    cout<<"rgbchannel[2] :"<<rgbchannel[2].size()<<endl;
    //model posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite runs from -1.0 ... +1.0
    //model multi_person_mobilenet_v1_075_float.tflite                 runs from  0.0 ... +1.0
    cout<<"image.channels() :"<<image.channels()<<endl;
    
    
     //waitKe(0); 
    //in=image.data;
    Len=image.rows*image.cols;
    for(i=0;i<Set.model_channels;i++)
    {   
        image=rgbchannel[i];
        in=image.data;
        for(int j=0;j<Len;j++)
        {
            f     =in[j];
            out[c]=(f-0) / 1;
            //cout<<"out"<<c<<":"<<out[c]<<endl;
            c++;
        }
        //waitKey(0); 	
    }
}

//Model卡片種類確認轉正 (-1:no Type)
int CardType_Rotate_model(Mat &frame,Mat &Card_Img,Mat &HSV_Card_Img, float temp_score,modelSet_t Set)
{
    int rModel;
    int CardType;
    float Tmpout=0;
   
    GetImageTFLite(interpreter->typed_tensor<float>(interpreter->inputs()[0]),frame,Set);
    
    
    cout << "run interpreter"<< std::endl;
    auto startTimer = std::chrono::steady_clock::now();
    // run interpreter
    if (interpreter->Invoke() != kTfLiteOk) {
    cout << "Failed to invoke!"<< std::endl;
    }
    auto endTimer = std::chrono::steady_clock::now();
    std::chrono::milliseconds t_msec = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer - startTimer);
    std::cout << "model t_msec = " << t_msec.count() << std::endl;
    
    auto output_data = interpreter->tensor(interpreter->outputs()[0])->data.f;
    
    for(int i=0;i<5;i++)
    {
        if(output_data[i]>temp_score)
        {
            if(Tmpout<output_data[i])
            {
                Tmpout=output_data[i];
                cout << "output_data:"<<output_data[i]<<endl;
                cout << "CardType:"<<i<<endl;
                rModel=i;
            }
            
        }
    }
    if(rModel==0||rModel==1)
    {
        //去朦朧
        auto startTimer_Dehaze = std::chrono::steady_clock::now();
        //Card_Img = dehaze(Card_Img);
        auto endTimer_Dehaze = std::chrono::steady_clock::now();
        std::chrono::milliseconds t_msec_Dehaze = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_Dehaze - startTimer_Dehaze);
        std::cout << "Dehaze msec = " << t_msec_Dehaze.count() << std::endl;
        
    }
   
    if(rModel==1||rModel==3)
    {
        cout << "180"<<endl;
        flip(Card_Img, Card_Img, -1);  //rotate 180
        flip(HSV_Card_Img, HSV_Card_Img, -1);  //rotate 180
        //imshow( "result_window", Card_Img );
    }

    if(rModel==0||rModel==1){
        return CardType=1;
    }else if(rModel==2||rModel==3){
        return CardType=0;
    }else{
        return CardType=-1;
    }
      
        
    // if(rModel==0||rModel==1) return CardType=1;
    // if(rModel==2||rModel==3) return CardType=0;
    // if(rModel==4) return CardType=-1;
    
    
}

//////////////////////////////////////////////////////
//OCR檢測位置框選*
int OCRDetect_integral(CardSet_t Card[],int CardType)
{
    Mat ID_img;
    Mat Name_img;

    cout << "=== Start Find Target on Card ==="<< endl;
    
    //0:HSV ,1:B_Channel,2:G_Channel,3:R_Channel
    //cout << "Card[CardType].OCR_Processing_type= "<< Card[CardType].OCR_Processing_type<< endl;
    if(Card[CardType].OCR_Processing_type==0)
    {
        // 健保卡
        cout << " -  Health IC Card" << endl;
        Mat hsv;
        Point ID_top(Card[CardType].ID_top_X,Card[CardType].ID_top_Y);
        Point ID_Down(Card[CardType].ID_Down_X,Card[CardType].ID_Down_Y);
        Point Name_top(Card[CardType].Name_top_X,Card[CardType].Name_top_Y);
        Point Name_Down(Card[CardType].Name_Down_X,Card[CardType].Name_Down_Y);
        Rect NameROI(Name_top,Name_Down);
        Rect IDROI(ID_top,ID_Down);

        /*** New ***/
        // Find ID
        cout << " -  Find ID Number..." << endl;
        Mat src_img;
        I_rgb_warped(IDROI).copyTo(src_img);
        
        cout << "Gray" << endl;
        Mat gray;
        cvtColor(src_img, gray, COLOR_BGR2GRAY);

        cout << "Threshold" << endl;
        threshold(gary, dst, 235, 255, THRESH_BINARY);

        Mat id_result;
        id_result = dst.clone();

        cout << "Erode" << endl;
        Mat erodeElement = getStructuringElement(MORPH_RECT, Size(20, 10));
        erode(dst, dst, erodeElement);

        cout << "Contours" << endl;
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(dst, contours, hierarchy,
                        RETR_CCOMP, CHAIN_APPROX_SIMPLE,Point(0, 0));

        if(contours.size()!=0){
            double max = -1;
            unsigned int max_index = 0;
            for(unsigned int i = 0;  i < contours.size();  i++){
                if (hierarchy[i][3] == -1)
                    continue;
                double area = contourArea(contours[i]);
                if(area>max){
                    max = area;
                    max_index = i;
                }
            }
        
            Rect rect = boundingRect(contours[max_index]);
            Point top(rect.x, rect.y);
            Point down(rect.br().x, rect.br().y);
            rectangle(src_img, top, down, Scalar(255, 0, 0), 1);
            
            Rect id_roi(top, down);
            id_result(id_roi).copyTo(IDimg);
        }
        cout << " -  ID Number Found." << endl;

        // Find Name
        cout << " -  Find Name..." << endl;
        I_rgb_warped(NameROI).copyTo(src_img);

        cout << "Gray" << endl;
        cvtColor(src_img, gray, COLOR_BGR2GRAY);

        cout << "Threshold" << endl;
        threshold(gray, dst, 193, 255, THRESH_BINARY);
        
        Mat name_result;
        name_result = dst.clone();
        
        cout << "Erode" << endl;
        erodeElement = getStructuringElement(MORPH_RECT, Size(20, 10));
        erode(dst, dst, erodeElement);
        
        cout << "Contours" << endl;
        contours.clear();
        rects.clear();
        hierarchy.clear();
        findContours(dst, contours, hierarchy,
                        RETR_CCOMP, CHAIN_APPROX_SIMPLE,Point(0, 0));

        if(contours.size()!=0){
            double max = -1;
            unsigned int max_index = 0;
            for(unsigned int i = 0; i < contours.size(); i++){
                if (hierarchy[i][3] == -1)
                    continue;
                double area = contourArea(contours[i]);
                if(area>max){
                    max = area;
                    max_index = i;
                }
            }
        
            Rect rect = boundingRect(contours[max_index]);
            Point top(rect.x, rect.y);
            Point down(rect.br().x, rect.br().y);
            rectangle(src_img, top, down, Scalar(255, 0, 0), 1);
            
            Rect name_roi(top, down);
            name_result(name_roi).copyTo(Nameimg);
        }
        cout << " -  Name Found." << endl;

        rectangle(I_rgb_warped, ID_top, ID_Down, cv::Scalar(0, 255, 0), 1);
        rectangle(I_rgb_warped, Name_top, Name_Down, cv::Scalar(0, 255, 0), 1);
        cout << "=== Find Target on Card End ==="<< endl;
        return 0;
    }
    else
    {
        // 身分證
        cout << " -  ID Card" << endl;
        Point ID_top(Card[CardType].ID_top_X,Card[CardType].ID_top_Y);
        Point ID_Down(I_rgb_warped.cols,I_rgb_warped.rows);
        Point Name_top(Card[CardType].Name_top_X,Card[CardType].Name_top_Y);
        Point Name_Down(Card[CardType].Name_Down_X,Card[CardType].Name_Down_Y);
        Rect NameROI(Name_top,Name_Down);
        Rect IDROI(ID_top,ID_Down);
        
        // ID 
        cout << " -  Find ID Number..." << endl;
        Mat src_img;
        I_rgb_warped(IDROI).copyTo(src_img);

        cout << "Gray" << endl;
        Mat dst;
        Mat gray;
        cvtColor(src_img, gray, COLOR_BGR2GRAY);

        /** Split R Channel **/
        vector<Mat> idcard_channels;
        split(src_img, idcard_channels);

        Mat &R_channel = idcard_channels[0];

        //imshow("RRRR", R_channel);
        //imshow("~~~R", ~R_channel);
        
        //3、二值化（降噪）
        cout << "th" << endl;
        Mat id_result;
        //threshold(R_channel, dst, 235, 255, THRESH_BINARY);
        // Test adaptive
        adaptiveThreshold(R_channel, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
        
        id_result = dst.clone();

        //imshow("threshold", dst);
        
        // 4.1 腐蝕、膨脹
        cout << "erode" << endl;
        Mat erodeElement = getStructuringElement(MORPH_RECT, Size(20, 10));
        erode(dst, dst, erodeElement);
        
        //imshow("erode", dst);

        //4、輪廓檢測，把所有的連續的閉包用矩形包起來
        cout << "contours" << endl;
        vector<vector<Point>> contours;
        vector<Rect> rects;

        //RotatedRect rect;
        vector<Vec4i> hierarchy;
        
        findContours(dst, contours, hierarchy,
                        RETR_CCOMP, CHAIN_APPROX_SIMPLE,Point(0, 0));

        if(contours.size()!=0){
            double max = -1;
            unsigned int max_index = 0;
            for(unsigned int i = 0;  i < contours.size();  i++){
                if (hierarchy[i][3] == -1)
                    continue;
                double area = contourArea(contours[i]);
                if(area>max){
                    max = area;
                    max_index = i;
                }
            }
        
            Rect rect = boundingRect(contours[max_index]);
            Point top(rect.x, rect.y);
            Point down(rect.br().x, rect.br().y);
            rectangle(src_img, top, down, Scalar(255, 0, 0), 1);
            
            //imshow("Final", src_img);
            
            Rect id_roi(top, down);
            //id_result(id_roi).copyTo(IDimg);
            gray(id_roi).copyTo(IDimg);
            threshold(IDimg, IDimg, 215, 255, THRESH_BINARY);
        }
        cout << " -  ID Number Found." << endl;
        
        // Test Name find //
        I_rgb_warped(NameROI).copyTo(src);
        //處理身份證
        src_img = src;   
        //2、灰度化
        cout << "Gray" << endl;
        cvtColor(src_img, dst, COLOR_BGR2GRAY);
        //imshow("GRAY", dst);
        
        //3、二值化（降噪）
        cout << "th" << endl;
        Mat name_result;
        threshold(dst, dst, 193, 255, THRESH_BINARY);
        name_result = dst.clone();
        //imshow("THRESHOLD", name_result);
        
        // 4.1 腐蝕、膨脹
        cout << "erode" << endl;
        erodeElement = getStructuringElement(MORPH_RECT, Size(20, 10));
        erode(dst, dst, erodeElement);
        
        //4、輪廓檢測，把所有的連續的閉包用矩形包起來
        cout << "contours" << endl;
        // vector<vector<Point>> contours;
        // vector<Rect> rects;

        //RotatedRect rect;
        // vector<Vec4i> hierarchy;
        contours.clear();
        rects.clear();
        hierarchy.clear();
        findContours(dst, contours, hierarchy,
                        RETR_CCOMP, CHAIN_APPROX_SIMPLE,Point(0, 0));

        if(contours.size()!=0){
            double max = -1;
            unsigned int max_index = 0;
            for(unsigned int i = 0;  i < contours.size();  i++){
                if (hierarchy[i][3] == -1)
                    continue;
                double area = contourArea(contours[i]);
                if(area>max){
                    max = area;
                    max_index = i;
                }
            }
        
            Rect rect = boundingRect(contours[max_index]);
            Point top(rect.x, rect.y);
            Point down(rect.br().x, rect.br().y);
            rectangle(src_img, top, down, Scalar(255, 0, 0), 1);
            
            //imshow("Line", src_img);
            
            Rect name_roi(top, down);
            name_result(name_roi).copyTo(Nameimg);
        }

        rectangle(I_rgb_warped, ID_top, ID_Down, cv::Scalar(0, 255, 0), 1);
        rectangle(I_rgb_warped, Name_top, Name_Down, cv::Scalar(0, 255, 0), 1);

        return 0;
    }
}

//OCR 
int OCR(CardSet_t Card[],int CardType, SysSet_t &Set,CvxText &tmp,modelSet_t modSet)
{       
        // Target img: Nameimg, IDimg
        //imshow("Nameimg",Nameimg);
        //imshow("IDimg",IDimg); 
        //while(1){ if(waitKey(0)==27) break;}
        char* str_name ;
        cout << " === OCR strat ==="<< endl;
        GetImageTFLite(interpreter_OCR->typed_tensor<float>(interpreter_OCR->inputs()[0]),Nameimg,modSet);
        //OCR Name
        cout << "run interpreter"<< std::endl;
        auto startTimer = std::chrono::steady_clock::now();
        // run interpreter
        if (interpreter_OCR->Invoke() != kTfLiteOk) {
        cout << "Failed to invoke!"<< std::endl;
        }
        auto endTimer = std::chrono::steady_clock::now();
        std::chrono::milliseconds t_msec = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer - startTimer);
        std::cout << "[Process Time]OCR Name: " << t_msec.count() << "msec." << std::endl;
    
        auto output_data = interpreter_OCR->tensor(interpreter_OCR->outputs()[0])->data.f;
        
        
        // fill in the inference result
        for (int i = 0; i < modSet.Label_dict; i++)
        {
            data[i].label_index = i;
            data[i].prob = output_data[i];
        }
    
        // sort the result (descending)
        qsort(data, modSet.Label_dict, sizeof(info_t), compare);
        
        //cout<<"output_data :"<<data[0].label_index<<" string: "<<Labels[data[0].label_index]<<" results:"<<data[0].prob<<endl;
        //cout<<"output_data :"<<data[1].label_index<<" string: "<<Labels[data[1].label_index]<<" results:"<<data[1].prob<<endl;
        //cout<<"output_data :"<<data[2].label_index<<" string: "<<Labels[data[2].label_index]<<" results:"<<data[2].prob<<endl;
        
        Text_name=Labels[data[0].label_index];
        cout << "[Result]Name String = " << Text_name << std::endl;


        startTimer = std::chrono::steady_clock::now();

        //OCR ID
        api_eng_id->SetImage((uchar*)IDimg.data, IDimg.size().width, IDimg.size().height, IDimg.channels(), IDimg.step1());
        api_eng_id->Recognize(0);
        char* outText_num = api_eng_id->GetUTF8Text();
      
        Text_num=string(outText_num);
        
        endTimer = std::chrono::steady_clock::now();
        t_msec = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer - startTimer);
        std::cout << "[Process Time]OCR ID: " << t_msec.count() << "msec." << std::endl;
        cout << "[Result]ID String = " << Text_num << std::endl;
        Text_num.erase(remove(Text_num.begin(), Text_num.end(), '\n'), Text_num.end());
        Text_num.erase(remove(Text_num.begin(), Text_num.end(), ' '), Text_num.end());
        //flip(frame, frame, 1);  //Flipped Horizontally 
        // DELETE \n
        if(Text_num.length()>10)
        { 
            if( Text_num[0]=='V') Text_num.erase(1,2);
            if( Text_num[0]=='Y') Text_num.erase(1,2);
            //ss.erase(remove(Text_num.begin(), Text_num.end(), '¥'), Text_num.end());
        }
            
        if(Text_num[0]=='0')Text_num[0]='O';
        if(Text_num[0]=='5')Text_num[0]='S';
        if(Text_num[0]=='9')Text_num[0]='S';
        if(Text_num[0]=='8')Text_num[0]='S';
        if(Text_num[0]=='6')Text_num[0]='G';
        if(Text_num[0]=='7')Text_num[0]='T';
        if(Text_num[0]=='3')Text_num[0]='J';
        if(Text_num[0]=='1')Text_num[0]='I';
        if(Text_num[0]=='2')Text_num[0]='Z';
        if(Text_num[0]=='[')Text_num[0]='L';
        if(Text_num[0]=='$')Text_num[0]='S';
        if(Text_num[0]=='4')Text_num[0]='A';
        
        //if(Text_num[0]=='¥')Text_num[0]='Y';
        //if(Text_num[0]=='£')Text_num[0]='E';
        //if(Text_num[0]=='€')Text_num[0]='E';
        //if(Text_num[0]=='©')Text_num[0]='O';
        //if(Text_num[0]=='§')Text_num[0]='S';
        
        //¥
        if(Text_num[0]+Text_num[1]==359)
        {
            Text_num[0]=' ';
            Text_num[1]='Y';
            Text_num.erase(remove(Text_num.begin(), Text_num.end(), ' '), Text_num.end());
        }
        //£
         if(Text_num[0]+Text_num[1]==357)
        {
            Text_num[0]=' ';
            Text_num[1]='E';
            Text_num.erase(remove(Text_num.begin(), Text_num.end(), ' '), Text_num.end());
        }
       //€
         if(Text_num[0]+Text_num[1]+Text_num[2]==528)
        {
            Text_num[0]=' ';
            Text_num[1]=' ';
            Text_num[2]='E';
            Text_num.erase(remove(Text_num.begin(), Text_num.end(), ' '), Text_num.end());
        }
       //©
         if(Text_num[0]+Text_num[1]==363)
        {
             Text_num[0]=' ';
             Text_num[1]='O';
             Text_num.erase(remove(Text_num.begin(), Text_num.end(), ' '), Text_num.end());
        }
        //§
          if(Text_num[0]+Text_num[1]==361)
        {
             Text_num[0]=' ';
             Text_num[1]='S';
             Text_num.erase(remove(Text_num.begin(), Text_num.end(), ' '), Text_num.end());
        }
        cout << "[OCR Final Result]\nName: " << Text_name << "\nID: " << Text_num << endl;
        cout << " === OCR End === "<< endl;
        return 0;
        
}
//OCR 結果比較

//RFID_Check
int RFID_Check(SysSet_t Set) 
{
    cout << "RFID Check"<< endl;
    ifstream input_file;
    vector<string> lines;
    string line;

    input_file.open(Set.RFIDPath);
    if (!input_file.is_open()) {       
        cout << "Could not open the file - '"<<Set.RFIDPath << "'" << endl;
    } 
    else 
    {
            getline(input_file, line);

            cout <<"lines.Size:"<<line.size()<< "  " << line << endl;
            CURR_RFID = line;
            if(line.size()!=0)
            {    
                line.erase(remove(line.begin(), line.end(), '\n'),line.end());
                line.erase(remove(line.begin(), line.end(), ' '),line.end());
          
                
          
                for(int i=0;i<_RFIDdata.size();i++)
                    {
                        if(_RFIDdata[i].RFID_UID==line)
                        {
                            rfid_Text_name=_RFIDdata[i].Name;
                   
                            rfid_Text_num=_RFIDdata[i].ID;
                            cout <<"RFID Name:"<<rfid_Text_name<< endl;
                            cout <<"RFID ID:"<<rfid_Text_num<< endl;
                            return 1;
                        }
            
                    }
             
            }
            else
            {
            cout <<"No Card"<< endl;
           
            input_file.close();
            return -1;
            }
        
       
    }
    input_file.close();
     
    
    return 0;
}

//SmartCard_Check
int SmartCard_Check(SysSet_t Set) 
{
    cout << "SmartCard Check"<< endl;
    ifstream input_file;
    vector<string> lines;
    string line;

    input_file.open(Set.SmartCardPath);
    if (!input_file.is_open()) {       
        cout << "Could not open the file - '"<<Set.SmartCardPath << "'" << endl;
    } 
    else 
    {
        while (getline(input_file, line)){
        lines.push_back(line);
        }
        cout <<"lines.Size:"<<lines.size()<< endl;
        if(lines.size()!=0)
        {
             cout <<"SmartCard Name:"<<lines[0]<< endl;
             Text_name=lines[0];
             cout <<"SmartCard ID:"<<lines[1]<< endl;
             Text_num=lines[1];
        }
        else
        {
            cout <<"No Card"<< endl;
           
            input_file.close();
            return -1;
        }
       
    }
    input_file.close();
     
    
    return 0;
}

//檢測結果傳送->i500
int SendOCR_result(SysSet_t &Set)
{
    cout << "SendOCR result strat "<< endl;
    // Command combine
    string Command="echo "+ Text_num +Set.sshPath;
    const char* Comd = Command.c_str();
    // send shh id data 
    cout << "send = " << Comd << endl;
    system(Comd);
    
    //Scp img 
    imwrite("OCR.png",r_frame);
    system("scp OCR.jpg root@192.168.53.32:/data");
    system("scp -o \"StrictHostKeyChecking no\" OCR.png root@192.168.50.229:/data");
    system("echo \"yes\"");
    
    cout << "SendOCR result end "<< endl;
    return 0;
}


bool acceptLinePair(Vec2f line1, Vec2f line2, float minTheta)
{
    float theta1 = line1[1], theta2 = line2[1];

    if(theta1 < minTheta)
    {
        theta1 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    if(theta2 < minTheta)
    {
        theta2 += CV_PI; // dealing with 0 and 180 ambiguities...
    }

    return abs(theta1 - theta2) > minTheta;
}

vector<Point2f> lineToPointPair(Vec2f line)
{
    vector<Point2f> points;

    float r = line[0], t = line[1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;

    points.push_back(Point2f(x0 + alpha*(-sin_t), y0 + alpha*cos_t));
    points.push_back(Point2f(x0 - alpha*(-sin_t), y0 - alpha*cos_t));

    return points;
}

// the long nasty wikipedia line-intersection equation...bleh...
Point2f computeIntersect(Vec2f line1, Vec2f line2)
{
    vector<Point2f> p1 = lineToPointPair(line1);
    vector<Point2f> p2 = lineToPointPair(line2);

    float denom = (p1[0].x - p1[1].x)*(p2[0].y - p2[1].y) - (p1[0].y - p1[1].y)*(p2[0].x - p2[1].x);
    Point2f intersect(((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].x - p2[1].x) -
                       (p1[0].x - p1[1].x)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom,
                      ((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].y - p2[1].y) -
                       (p1[0].y - p1[1].y)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom);

    return intersect;
}


int main(int argc, char *argv[])
{
    initSystem(OCRfile_Name,Seting,Card);
    initTesseract(Seting);
    
    //顯示中文設定
    CvxText text("./kaiu.ttf"); //指定字體
    Scalar size1{ 50, 0.5, 0.1, 0 }; // (字體大小, 無效的, 字符間距, 無效的 }
    text.setFont(nullptr, &size1, nullptr, 0);
    
    //CamSet
    VideoCapture cap(-1);
    cap.set(CAP_PROP_FRAME_WIDTH,Seting.cap_width);
    cap.set(CAP_PROP_FRAME_HEIGHT,Seting.cap_height);
    cap.set(CAP_PROP_BUFFERSIZE,1);
    Rect FoVRect(Seting.Fov_X,Seting.Fov_Y,Seting.Fov_W,Seting.Fov_H);
    
    //DataSender API
    cgmhDataSender sender;
    
    //TFLife init     
    cout << "[System State]Initial Card Recognition Model" << endl;
    init_TFLife_model(Seting._tflife_path,Model_Set,interpreter,model,resolver);
    
    //TF OCR init   
    cout << "[System State]Initial Name Recognition Model" << endl;
    init_TFLife_model(Seting.OCR_Chi,OcRModel_Set,interpreter_OCR,model_OCR,resolver_OCR);
    ifstream in("tch_name_dict_1685.txt");
    if(!in.is_open()) return -1;
    string str;
    while(getline(in, str)){
        if(str.size()>0) Labels.push_back(str);
    }
    in.close();
    //Label_dict
    OcRModel_Set.Label_dict=Labels.size();
    //getRFIDdata
    getRFIDdata("RFIDdata.txt");
    
    // ------------------------------------------------------------------------
    // GUI
    // ------------------------------------------------------------------------
    if(Seting.show_r) cvui::init(WINDOW_NAME);
    
    //cvui::init(WINDOW1_NAME);
    int x = 660;
    int width = 450;
    Mat WinFrom = cv::Mat(550, 660, CV_8UC3);
    Mat OCR_img=Mat(250, 768, CV_8UC3);
    OCR_img=Scalar(49, 52, 49);
    Mat Loading_img=Mat(250, 768, CV_8UC3);
    Loading_img=Scalar(49, 52, 49);
    Mat Warn_img=Mat(250, 768, CV_8UC3);
    Warn_img=Scalar(49, 52, 49);
    Mat instruction=imread("instruction.png");  
    namedWindow("OCR Result");
    moveWindow("OCR Result", 0, 1200);
    setWindowProperty("OCR Result",WND_PROP_TOPMOST,WINDOW_GUI_NORMAL);
    
    if(Seting.bUse_RFID||Seting.bUse_SmartCard)
    {
        RFID_img=Scalar(49, 52, 49);
        namedWindow("Smart Card Result");
        moveWindow("Smart Card Result", 0, 1200);
        setWindowProperty("Smart Card Result",WND_PROP_TOPMOST,WINDOW_GUI_NORMAL);  
    }
    
    cout << "[System State]Initial UI" << endl;
    
    while(bOCR)
    {    
        // check read frame succeeded
        if (!cap.read(frame)) {
            cerr << "ERROR! blank frame grabbed\n";
            continue;
        }
        else
        {
            //offline
            if(Seting.offline)
            {
                frame=imread(Seting.off_ImgePath);  
            }
                
            auto ALLstartTimer = std::chrono::steady_clock::now();
                
            bCardDetect=false;
            Type_num =-1;
            rRFID=-1;
            d1=0;
            d2=0;
                
            ////////RGB Find Card////////////////////////////////////////
            auto startTimer_FindCard = std::chrono::steady_clock::now();

            Mat rframeImg;
            frame(FoVRect).copyTo(FoV_img);
            resize(FoV_img,rframeImg,Size(FoV_img.cols/10,FoV_img.rows/10),INTER_NEAREST);
            cvtColor(rframeImg,hsv,COLOR_BGR2HSV);//轉成hsv平面
            inRange(hsv,Scalar(Seting.H_min,Seting.S_min,Seting.V_min) , Scalar(Seting.H_max,Seting.S_max,Seting.V_max), r_hsv); 
            Mat mask=Mat::zeros(rframeImg.rows,rframeImg.cols, CV_8U); //為了濾掉其他顏色
            mask=r_hsv;
            frame.copyTo(dst); //將原圖片經由遮罩過濾後，得到結果dst                                    
            Mat Processing;
            //Canny Edge Detection
            Canny(r_hsv, Processing, 0, 50, 3,true);
            
            RotatedRect resultRect;
            vector<Vec4i> hierarchy;
            vector<vector<Point>> contours;
            findContours(Processing, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
            if(contours.size()!=0){
                double max = -1;
                unsigned int max_index = 0;
                for(unsigned int i = 0;  i < contours.size();  i++){
                    double area = contourArea(contours[i]);
                    if(area>max){
                        max = area;
                        max_index = i;
                    }
                }
                
                resultRect = minAreaRect(contours[max_index]);//獲取輪廓的最小外接矩形
                Point2f pt[4];
                resultRect.points(pt);//獲取最小外接矩形的四個頂點座標
                //繪製最小外接矩形
                line(rframeImg, pt[0], pt[1], Scalar(255, 0, 0), 2, 8);
                line(rframeImg, pt[1], pt[2], Scalar(255, 0, 0), 2, 8);
                line(rframeImg, pt[2], pt[3], Scalar(255, 0, 0), 2, 8);
                line(rframeImg, pt[3], pt[0], Scalar(255, 0, 0), 2, 8);
                cout << "*******************************" << endl;
                cout << "X座標" << resultRect.center.x << "Y座標" << resultRect.center.y << "偏移角度"<<resultRect.angle<<endl;
                cout << "*******************************" << endl;
                //imshow("Line", rframeImg);
            }

            // Ken
            auto startTimer_warped = std::chrono::steady_clock::now();
            resultRect.points(sp);
            sp[0] =sp[0]*10;
            sp[1] =sp[1]*10;
            sp[2] =sp[2]*10;
            sp[3] =sp[3]*10;
            sort(sp,sp+4,comp);
            d1 = norm(sp[0]-sp[1]);
            d2 = norm(sp[0]-sp[2]);
            cout << "sp[0] = " << sp[0] << endl;
            cout << "sp[1] = " << sp[1] << endl;
            cout << "sp[2] = " << sp[2] << endl;
            cout << "sp[3] = " << sp[3] << endl;
            cout << "d1 = " << d1 << endl;
            cout << "d2 = " << d2 << endl;
            
            int bw = int(resultRect.size.width)*10;
            int bh = int(resultRect.size.height)*10;
            if (bw < bh)
            {
                int tmp = bw;
                bw = bh;
                bh = tmp;
            }
            Mat M;
            if(sp[0].x>sp[1].x)
            {
                Point2f dp[] = { Point2f(bw-1, 0), Point2f(0, 0),Point2f(bw-1, bh-1), Point2f(0, bh-1) };
                M = getPerspectiveTransform(sp, dp);
            }
            else
            {
                Point2f dp[] = {Point2f(0, 0), Point2f(bw-1, 0),Point2f(0, bh-1), Point2f(bw-1, bh-1)};
                M = getPerspectiveTransform(sp, dp);
            }

            //INTER_NEAREST test....
            //warpPerspective(dst,I_hsv_warped, M, Size(bw, bh));

            warpPerspective(FoV_img, I_rgb_warped, M, Size(bw, bh));
        
            auto endTimer_warped = std::chrono::steady_clock::now();
            std::chrono::milliseconds t_msec_warped = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_warped - startTimer_warped);
            std::cout << "Find contours and warped computeIntersect = " << t_msec_warped.count() << std::endl;

            // imshow("I_rgb_warped",I_rgb_warped);
            // imshow("Fov", FoV_img);
            //imshow("Processing", Processing);
            //waitKey(0);
            auto endTimer_FindCard = std::chrono::steady_clock::now();
            std::chrono::milliseconds t_msec_HoughLines = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_FindCard - startTimer_FindCard);
            std::cout << "Find Card t_msec = " << t_msec_HoughLines.count() << std::endl;
            ////////RGB Find Card End////////////////////////////////////////

            //RFID Card//////////////////////////////////////////////////////
            if(Seting.bUse_RFID)
            {
                rRFID=RFID_Check(Seting);
                if(rRFID==-1)
                {
                    moveWindow("Smart Card Result", 0, 1200);
                    bRFIDOK=false;
                    bRFIDNG=false;
                    RFID_send=true;
                    rRFID=-1;
                    
                }
                else 
                {
                    cout << "CRFID" << CURR_RFID << endl;
                    if(rfid_Text_num=="") bRFIDNG=true;
                    else bRFIDOK=true;

                }
                
            }
            
                //Smart Card////////////////////////////////////////////////////  
            if(Seting.bUse_SmartCard&&rRFID==-1)
            {
                rRFID=SmartCard_Check(Seting);
                if(rRFID==-1)
                {
                    moveWindow("Smart Result", 0, 1200);
                    bRFIDOK=false;
                    RFID_send=true;
                }
                else 
                {
                    bRFIDOK=true;
                }
            }
            
            
            if(CURR_RFID=="START") {
                bRFIDOK=false;
                bRFIDNG=false;
            }
            
            
            //卡片種類辨識、旋轉/////////////////////////////////////////////
            auto startTimer_CardType = std::chrono::steady_clock::now();
            if(d2>Seting.Card_PixY&&d1>Seting.Card_PixX&&d2<1000&&d1<2000||d1>Seting.Card_PixY&&d2>Seting.Card_PixX&&d2<2000&&d1<1000)
            {
                cout << "[Card]Find Card Pixel:" << d1 << ", " << d2 << ")" <<endl;
                int rType=-1;              
                //辨識卡片種類結果   
                rType=CardType_Rotate_model(frame,I_rgb_warped,I_hsv_warped,Card[0].TempScore,Model_Set);
                if(rType !=-1)
                {
                    bCardDetect=true;
                    Type_num =rType;
                    cout << "[Card]Card Type:" << Type_num <<endl;
                    if(bOCR_End)
                    {
                        TimeOutStart = std::chrono::steady_clock::now();
                        bOCR_Start=true;
                        bOCR_End=false;  
                        cout << " -> OCR_Start" <<endl;
                    }
                } 
                if(rType ==-1&&bOCR_End)
                {
                    cout << "[Card]Not IC or ID Card!" <<endl;  
                    bOCR_Start=false;
                    bCardDetect = false;
                    bOCRNG=true; 
                }
            }

            //護照判別
            if (Seting.bUse_PassPort)
            {
                cout << "Passport D1:" << d1 << endl;
                if (d1 > Seting.PassPort_PixX)
                {
                    Mat port;
                    Point Port_pt1, Port_pt2;
                    
                    Port_pt1 = sp[0];
                    Port_pt2 = sp[1];
                    Port_pt2.y = FoV_img.rows;
                    
                    Rect  PortRect(Port_pt1, Port_pt2);
                    FoV_img(PortRect).copyTo(port);
                    imshow("PassPort_rgb_", port);
                    //imshow("HSV", r_hsv);
                    //imshow("Processing", Processing);
                    //if (waitKey(0)) break;	
                }
            }
            

            auto endTimer_CardType = std::chrono::steady_clock::now();
            std::chrono::milliseconds t_msec_CardType = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_CardType - startTimer_CardType);
            std::cout << "[Process Time]Find Card: " << t_msec_CardType.count() << "msec." << std::endl;
            
            cout << "===== DEBUG =====\n  d1:" << d1 << ", d2:" << d2 << endl;
            cout << "  bCardDetect:" << bCardDetect << ", bOCR_Start:" << bOCR_Start << ", bOCROK:" << bOCROK << endl;
            if(d1>200&&d2>200&&!bCardDetect&&!bOCR_Start&&!bOCROK)
            {
                cout << "Check Position" << endl;
                bWarn=true;
            }
            
            //開始OCR辨識
            auto startTimer_OCRALL = std::chrono::steady_clock::now();
            
            if(bCardDetect&&bOCR_Start)
            {
                int rOCR_Detect;
                
                //OCR ID Name ROI bounding
                auto startTimer_OCRDetect = std::chrono::steady_clock::now();
                cout << "OCR Type" << Type_num << endl;
                rOCR_Detect=OCRDetect_integral(Card, Type_num);
                //imshow("I_hsv_warped",I_hsv_warped);
                auto endTimer_OCRDetect = std::chrono::steady_clock::now();
                std::chrono::milliseconds t_msec_OCRDetect = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_OCRDetect - startTimer_OCRDetect);
                std::cout << "[Process Time]OCR Find Target:" << t_msec_OCRDetect.count() << "msec." << std::endl;
                //imshow("I_rgb_warped",I_rgb_warped);
                //OCR
                if(rOCR_Detect==-1)
                {
                    cout << " -> OCR Detect NG" <<endl;  
                    bOCR_Start=false;
                    bOCRNG=true;
                }
                else
                {
                    auto startTimer_OCR = std::chrono::steady_clock::now();
                    cout << " -> Start Recognition" << endl;
                    OCR(Card,Type_num,Seting,text,OcRModel_Set);
                    auto endTimer_OCR = std::chrono::steady_clock::now();
                    std::chrono::milliseconds t_msec_OCR = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_OCR - startTimer_OCR);
                    std::cout << "[Process Time]OCR: " << t_msec_OCR.count() << "msec." << std::endl;
                    OCRresult_t OCR_R;
                    OCR_R.Name=Text_name;
                    OCR_R.ID=Text_num;
                    _OCRresult.push_back(OCR_R);
                    //cout << "_OCRresult.size()" << _OCRresult.size() <<endl;   
                }      
                
                //OCR 3次後比較 
                if(_OCRresult.size()>=3)
                {       
                    if(_OCRresult[0].Name==_OCRresult[1].Name)
                        Text_name=_OCRresult[0].Name;
                    else if(_OCRresult[0].Name==_OCRresult[2].Name)
                        Text_name=_OCRresult[0].Name;
                    else if(_OCRresult[1].Name==_OCRresult[2].Name)
                        Text_name=_OCRresult[1].Name;
                    else
                        Text_name="";
                    
                    if(_OCRresult[0].ID==_OCRresult[1].ID)
                        Text_num=_OCRresult[0].ID;
                    else if(_OCRresult[0].ID==_OCRresult[2].ID)
                        Text_num=_OCRresult[0].ID;
                    else if(_OCRresult[1].ID==_OCRresult[2].ID)
                        Text_num=_OCRresult[1].ID;
                    else
                        Text_num="";
                        
                    if (Text_name.empty() && Text_num.empty()){
                        bOCR_Start=false;    
                        bOCROK=true;
                        cout << "_OCRresult[0]"<<_OCRresult[0].Name <<endl; 
                        cout << "_OCRresult[1]"<<_OCRresult[1].Name <<endl; 
                        cout << "_OCRresult[2]"<<_OCRresult[2].Name <<endl; 
                    }else{
                        bOCRNG = true;
                    }
                    

                }
                else if(_OCRresult.size()<3&&rOCR_Detect!=-1)
                {
                    auto startTimer_Wait = std::chrono::steady_clock::now();
                    string Text_load="卡片資料讀取中"
                    for(int i = 0; i < _OCRresult.size(); i++){
                        Text_load += ".."
                    }
                    char* c_load=(char*)Text_load.c_str();
                    wchar_t* w_load;
                    ToWchar(c_load, w_load);
                    text.putText(Loading_img, w_load, cv::Point(10, 50), cv::Scalar(255, 255, 255));
                    imshow("OCR Result",Loading_img);
                    moveWindow("OCR Result", 0, 750);
                    auto endTimer_Wait = std::chrono::steady_clock::now();
                    std::chrono::milliseconds t_msec_Wait = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_Wait - startTimer_Wait);
                    std::cout << "Wait msec = " << t_msec_Wait.count() << std::endl;
                }
            }
            else
            {
                
                cout << "CardDetect ,  OCR_Start= "<< bCardDetect<<","<< bOCR_Start << endl;
            }
            auto endTimer_OCRALL = std::chrono::steady_clock::now();
            std::chrono::milliseconds t_msec_OCRALL = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_OCRALL - startTimer_OCRALL);
            std::cout << "[Process Time]Full OCR: " << t_msec_OCRALL.count() << "msec." << std::endl;
            
            
            //Confirmation timed out
            if(bOCR_Start)
            {
                TimeOutEnd = std::chrono::steady_clock::now();
                std::chrono::milliseconds TimeOut = std::chrono::duration_cast<std::chrono::milliseconds>(TimeOutEnd - TimeOutStart);
                std::cout << "Time Out count: = " << TimeOut.count() << std::endl;
                if(TimeOut.count()>4000)
                {
                    bOCRNG=true; 
                }
            }
            cout << " === Reset Check === " << endl;
            cout << " d1:" << d1 << ", d2:" << d2 << "\nCard Type:" << Type_num << ", RFIDOK:" << bRFIDOK << endl;
            //卡片取走初始化
            if(d2<=60&&d1<=60&&Type_num==-1)
            {
                bCardDetect=false;
                bOCR_Start=false;
                bOCR_End=true;
                bOCRNG=false; 
                bOCROK=false; 
                OCR_img=Mat(250, 768, CV_8UC3);
                OCR_img=Scalar(49, 52, 49);
                // RFID_img=Mat(250, 768, CV_8UC3);
                // RFID_img=Scalar(49, 52, 49);
                Loading_img=Mat(250, 768, CV_8UC3);
                Loading_img=Scalar(49, 52, 49);
                Text_name="";
                Text_num="";
                rfid_Text_name="";
                rfid_Text_num="";
                rRect= RotatedRect(Point2f(0,0), Size2f(0,0), 0);
                cout << "[System State]Reset OCR" <<endl;
                _OCRresult.clear();
                moveWindow("OCR Result", 0, 1200);
                //moveWindow("Loading card data...", 0, 1200);
                            
            }

            if(!bRFIDOK || (CURR_RFID!="" && CURR_RFID != PREV_RFID)){
            
                RFID_img=Mat(250, 768, CV_8UC3);
                RFID_img=Scalar(49, 52, 49);
                
                cout << "[RFID]Start->End" <<endl;
                PREV_RFID = CURR_RFID;
                //moveWindow("Smart Card Result", 0, 1200);
                //moveWindow("Loading card data...", 0, 1200);
            }
            
            
            if(bWarn)
            {
                cout << "======Warn======"<< endl;
                cout << "instruction.size()"<<instruction.size()<< endl;
                string Text_load="請檢查 卡片";
                
                char* c_load=(char*)Text_load.c_str();
                wchar_t* w_load;
                ToWchar(c_load, w_load);
                text.putText(Warn_img, w_load, cv::Point(10, 60), cv::Scalar(255, 255, 255));
                
                string Text_load1=" 放置位置";
                char* c_load1=(char*)Text_load1.c_str();
                wchar_t* w_load1;
                ToWchar(c_load1, w_load1);
                text.putText(Warn_img, w_load1, cv::Point(10, 110), cv::Scalar(255, 255, 255));
                
                Mat RoiImg, GrayImg, MaskImg, InvMaskImg;
                cvtColor(instruction, GrayImg,COLOR_BGR2GRAY);
                threshold(GrayImg, MaskImg, 0, 255, THRESH_BINARY);
                bitwise_not(MaskImg, InvMaskImg);
                InvMaskImg=255-InvMaskImg;
                RoiImg = Warn_img(Rect(400, 10,  instruction.cols,  instruction.rows));
                instruction.copyTo(RoiImg, InvMaskImg);
                
                imshow("OCR Result",Warn_img);
                moveWindow("OCR Result", 0, 750);
                bWarn=false;
            }   
            
                
            
            //OCR OK/NG顯示    
            auto startTimer_OCROKNG = std::chrono::steady_clock::now();    
            if(bOCROK && !Text_name.empty() && !Text_num.empty())
            {
                // Try to change result picture
                resize(I_rgb_warped, r_frame, Size(320,240), INTER_NEAREST);
                //resize(frame,r_frame , Size(320,240),INTER_NEAREST);
                
                rText_name="MR./MS. : "+Text_name.substr(0,3);
                char* cname=(char*)rText_name.c_str();
            
                wchar_t* wname;
                ToWchar(cname, wname);
                text.putText(OCR_img, wname, cv::Point(10, 50), cv::Scalar(255, 255, 255));
            
            
                rText_num="ID: "+Text_num;
                char* cID=(char*)rText_num.c_str();
                wchar_t* wID;
                ToWchar(cID, wID);
                text.putText(OCR_img, wID, cv::Point(10, 120), cv::Scalar(255, 255, 255));
                
                string rstr="請 收回證件";
                char* cstr=(char*)rstr.c_str();
                wchar_t* wstr;
                ToWchar(cstr, wstr);
                text.putText(OCR_img, wstr, cv::Point(10, 190), cv::Scalar(255, 255, 255));
                
                //moveWindow("Loading card data...", 0, 1200); 
                Mat RoiImg, GrayImg, MaskImg, InvMaskImg;
                moveWindow("OCR Result", 0, 750);                    
                cvtColor(r_frame, GrayImg,COLOR_BGR2GRAY);
                threshold(GrayImg, MaskImg, 0, 255, THRESH_BINARY);
                bitwise_not(MaskImg, InvMaskImg);
                InvMaskImg=255-InvMaskImg;
                RoiImg = OCR_img(Rect(400, 10, r_frame.cols, r_frame.rows));
                r_frame.copyTo(RoiImg, InvMaskImg);
                imshow("OCR Result",OCR_img);
                string Data=getCurrentSystemTime(); 
                //cout << "Data: "<<Data << endl;
                //waitKey(0);
                if(Seting.send_commd)
                {
                cout << "SendOCR_result " << endl;
                //string i500Ip, string fileDst
                sender.getI500Data(Seting.SCPRath, "/data/i500result.txt");
                //string lastName, string timeStamp, string personId
                cout << "lastName: "<<Text_name << endl;
                cout << "timeStamp: "<<Data << endl;
                cout << "personId: "<<Text_num << endl;
                sender.genData(Text_name, Data, Text_num);
                //cv::Mat faceFrame, cv::Mat idCardFrame
                sender.writeRequiredImg(I_rgb_warped, I_hsv_warped);
                //string cgmhUrl
                sender.sendCgmhData(Seting.CgmhUrl);
                //sender.sendCgmhDataTEST(Seting.CgmhUrl,"A123456789_data.json");
                //SendOCR_result(Seting);
                }
                
                
                // bOCROK=false;
            } 
            if(bOCRNG)
            {
                Loading_img=Mat(250, 768, CV_8UC3);
                Loading_img=Scalar(49, 52, 49);
                string Text_load="請重新放置卡片";
                char* c_load=(char*)Text_load.c_str();
                wchar_t* w_load;
                ToWchar(c_load, w_load);
                text.putText(Loading_img, w_load, cv::Point(10, 50), cv::Scalar(255, 255, 255));
                imshow("OCR Result",Loading_img);
                moveWindow("OCR Result", 0, 750);
                if(Seting.bSave_Img)
                {
                    cout << "Save NG Data"<< endl;
                    string Data=getCurrentSystemTime(); 
                    if(!dst.empty()); 
                    imwrite("./NG/"+Data+"_src.jpg",dst);
                }
                
            }
            
            auto endTimer_OCROKNG = std::chrono::steady_clock::now();
            std::chrono::milliseconds t_msec_OCROKNG = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_OCROKNG - startTimer_OCROKNG);
            std::cout << "OCR OK/NG msec = " << t_msec_OCROKNG.count() << std::endl;
            //RFID OK
            cout << "RFIDOK?" << bRFIDOK << endl;
            if(bRFIDOK&&CURR_RFID!="START")
            {
                //ifstream inFile;
                //inFile.open("i500result.txt"); //open the input file
                //stringstream strStream;
                //strStream << inFile.rdbuf(); //read the file
                //string str = strStream.str(); //str holds the content of the file
                //str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
                //string delimiter = ",";
                //string _temperature = str.substr(str.find(delimiter) + 1 , -1); 
                //inFile.close();
                
                //resize(frame,r_frame , Size(320,240),INTER_NEAREST);
                //putText(RFID_img, _temperature, cv::Point(400, 50),cv::FONT_HERSHEY_COMPLEX,2, cv::Scalar(0, 255, 0),3);
                rfid_text_name="MR./MS. :"+rfid_Text_name;
                std::cout << rfid_text_name << std::endl;
                char* cname=(char*)rfid_text_name.c_str();

                wchar_t* wname;
                ToWchar(cname, wname);
                text.putText(RFID_img, wname, cv::Point(10, 50), cv::Scalar(255, 255, 255));
            
            
                rfid_text_id="ID: "+rfid_Text_num;
                std::cout <<  rfid_text_id << std::endl;
                char* cID=(char*)rfid_text_id.c_str();
                wchar_t* wID;
                ToWchar(cID, wID);
                text.putText(RFID_img, wID, cv::Point(10, 120), cv::Scalar(255, 255, 255));
                
                string rstr="請 收回證件";
                char* cstr=(char*)rstr.c_str();
                wchar_t* wstr;
                ToWchar(cstr, wstr);
                text.putText(RFID_img, wstr, cv::Point(10, 190), cv::Scalar(255, 255, 255));
                
                //moveWindow("Loading card data...", 0, 1200); 
                Mat RoiImg, GrayImg, MaskImg, InvMaskImg;
                moveWindow("Smart Card Result", 0, 750);                    
                imshow("Smart Card Result",RFID_img);
                string Data=getCurrentSystemTime(); 
                cout << "Data: "<<Data << endl;
                //waitKey(0);
                if(Seting.send_commd&&RFID_send)
                {
                    cout << "SendOCR_result " << endl;
                    //SendOCR_result(Seting);
                    RFID_send=false;
                }
            }
            if(bRFIDNG)
            {
                Loading_img=Mat(250, 768, CV_8UC3);
                Loading_img=Scalar(49, 52, 49);
                string Text_load="卡片 未登入";
                char* c_load=(char*)Text_load.c_str();
                wchar_t* w_load;
                ToWchar(c_load, w_load);
                text.putText(Loading_img, w_load, cv::Point(10, 50), cv::Scalar(255, 255, 255));
                imshow("Smart Card Result",Loading_img);
                moveWindow("Smart Card Result", 0, 750);
            }
            
            auto startTimer_showUI = std::chrono::steady_clock::now(); 
                
            if(Seting.show_r)
            {
                
                
                Mat r_rgb_warped;
                resize(frame,r_frame , Size(320,240),INTER_NEAREST);
                if(!I_rgb_warped.empty())
                resize(I_rgb_warped,r_rgb_warped , Size(320,240),INTER_NEAREST);
                
                rText_name="Name: "+Text_name;
                char* cname=(char*)rText_name.c_str();
                
                wchar_t* wname;
                ToWchar(cname, wname);
                //text.putText(OCR_img, wname, cv::Point(10, 50), cv::Scalar(255, 255, 255));
                
                
                rText_num="ID: "+Text_num;
                char* cID=(char*)rText_num.c_str();
                wchar_t* wID;
                ToWchar(cID, wID);
                //text.putText(OCR_img, wID, cv::Point(10, 120), cv::Scalar(255, 255, 255));
                
                
                if (Seting.bSeting) 
                {   
                    WinFrom = cv::Mat(1024, 1150, CV_8UC3); 
                    WinFrom = cv::Scalar(49, 52, 49);
                    //cvui::image(WinFrom, 24, 10, r_RTSPframe);
                    cvui::image(WinFrom, 10, 10, r_frame);
                    cvui::image(WinFrom, 24, 480+50, OCR_img);
                    cvui::checkbox(WinFrom, 24, 480 + 30, "Parameter setting", &Seting.bSeting, 0xff9912);
                    //cvui::checkbox(WinFrom, 300, 480 + 30, "Save setting", &Card.bSave, 0xff9912);
                    // Render the settings window to house the UI
                    cvui::window(WinFrom, x, 10, 480, 480, "Parameter setting");
                    //cvui::text(WinFrom , x, 40, "Parameter setting", 0.6, 0X00ff00);
                    cvui::text(WinFrom , x, 40, "H_min", 0.4);
                    cvui::trackbar(WinFrom , x+10, 50, width, &Seting.H_min, 0, 180, 15);
                    cvui::text(WinFrom , x, 100, "H_max", 0.4);
                    cvui::trackbar(WinFrom , x+10, 110, width, &Seting.H_max, 0, 180, 15);
                    cvui::text(WinFrom , x, 160, "S_min", 0.4);
                    cvui::trackbar(WinFrom , x+10, 170, width, &Seting.S_min, 0, 255, 15);
                    cvui::text(WinFrom , x, 220, "S_max", 0.4);
                    cvui::trackbar(WinFrom , x+10, 230, width, &Seting.S_max, 0, 255, 15);
                    cvui::text(WinFrom , x, 280, "V_min", 0.4);
                    cvui::trackbar(WinFrom , x+10, 290, width, &Seting.V_min, 0, 255, 15);
                    cvui::text(WinFrom , x, 340, "V_max", 0.4);
                    cvui::trackbar(WinFrom , x+10, 350, width, &Seting.V_max, 0, 255, 15);
                    
                    if (!r_rgb_warped.empty())
                    cvui::image(WinFrom, 340, 10, r_rgb_warped);
                        
                    Mat r_Processing,r_Nameimg,r_IDimg;
                    cvtColor(Processing,r_Processing,COLOR_GRAY2BGR);//轉成hsv平面 
                    
                    cvui::image(WinFrom, 10, 260, rframeImg);
                    cvui::image(WinFrom, 340, 260,r_Processing);
                    // cvui::image(WinFrom, 10,360, r_Nameimg);
                    //cvui::image(WinFrom, 340, 360, r_IDimg);
                    if (!Nameimg.empty())
                    {
                        cvtColor(Nameimg, r_Nameimg, COLOR_GRAY2BGR);
                        cvui::image(WinFrom, 10, 360, r_Nameimg);
                    }
                    if (!IDimg.empty())
                    {
                        cvtColor(IDimg, r_IDimg, COLOR_GRAY2BGR);
                        cvui::image(WinFrom, 340, 360, r_IDimg);
                    }
                    //namedWindow("I_rgb_warped",1);
                    //setMouseCallback("I_rgb_warped",onMouse,NULL);

                    
                    if (r_hsv.empty())
                    {
                        cout <<"OCR frame didn't get anything yet"<< endl;
                        continue;
                    }  
                    imshow("HSV_frame",r_hsv);
                    }
                else
                {
                    //imshow("OCR_img",r_frame);
                    WinFrom = cv::Mat(1024, 768, CV_8UC3); 
                    WinFrom = cv::Scalar(49, 52, 49); 
                    
                    //cvui::image(WinFrom, 24, 480+50, OCR_img);
                    cvui::checkbox(WinFrom, 24, 480 + 30, "Parameter setting", &Seting.bSeting, 0xff9912);
                    //cvui::checkbox(WinFrom, 300, 480 + 30, "Save setting", &Card.bSave, 0xff9912);
        
                    cvui::image(WinFrom, 10, 10, r_frame);
                    if(!I_rgb_warped.empty())
                    cvui::image(WinFrom, 340, 10, r_rgb_warped);
                    
                    Mat r_Processing,r_Nameimg,r_IDimg;
                    cvtColor(Processing,r_Processing,COLOR_GRAY2BGR);//轉成hsv平面 
                    
                    
                    cvui::image(WinFrom, 10, 260, rframeImg);
                    cvui::image(WinFrom, 340, 260,r_Processing);
                    if(!Nameimg.empty())
                    {
                            cvtColor(Nameimg,r_Nameimg,COLOR_GRAY2BGR);
                            cvui::image(WinFrom, 10,360, r_Nameimg);
                    }
                    if(!IDimg.empty())
                    {
                            cvtColor(IDimg,r_IDimg,COLOR_GRAY2BGR);
                            cvui::image(WinFrom, 340, 360, r_IDimg);
                    }
                    
                    
                    //cvui::image(WinFrom, 24, 10,r_RTSPframe);
                    cvui::text(WinFrom , 10, 480+60, rText_num, 0.4);
                    
                    
                    //測試存資料
                    if (cvui::button(WinFrom, 24, 555, "TEST SENDING"))
                    {
                        
                        
                        //Mat testimage;
                        //testimage=imread("Test.jpg");
                        cout << "bottom SendOCR result !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
                        //string i500Ip, string fileDst
                        sender.getI500Data(Seting.SCPRath, "/data/i500result.txt");
                        //string lastName, string timeStamp, string personId
                        string Data=getCurrentSystemTime(); 
                        sender.genData("TestLee", Data, "A123456789");
                        //cv::Mat faceFrame, cv::Mat idCardFrame
                        sender.writeRequiredImg(frame, frame);
                        //string cgmhUrl
                        sender.sendCgmhData(Seting.CgmhUrl);
                        
                    }                     
                    //測試存資料en
                    
                    }
                    cvui::update();
                    cvui::imshow(WINDOW_NAME, WinFrom);
                
            }
            auto endTimer_showUI = std::chrono::steady_clock::now();
            std::chrono::milliseconds t_msec_showUI = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer_showUI - startTimer_showUI);
            std::cout << "show UI msec = " << t_msec_showUI.count() << std::endl;
            
            if(waitKey(150)==(char)115)//s
            {
                string Data=getCurrentSystemTime(); 
                cout << "./Snap/"+Data+"_src.jpg" << endl; 
                if(!dst.empty())
                imwrite("./Snap/"+Data+"_src.jpg",dst);
                if(!I_rgb_warped.empty())
                imwrite("./Snap/"+Data+"_rgb.jpg",I_rgb_warped);
                if(!I_hsv_warped.empty())
                imwrite("./Snap/"+Data+"_hsv.jpg",I_hsv_warped);
                if(!Nameimg.empty())
                imwrite("./Snap/"+Data+"_Name.jpg",Nameimg);
                if(!IDimg.empty())
                imwrite("./Snap/"+Data+"_ID.jpg",IDimg);
                cout << "===Save Image=== " << endl;
            }
            if (waitKey(5)==(char)113)
                break;
                
            
            auto ALLendTimer = std::chrono::steady_clock::now();
            std::chrono::milliseconds ALL_t_msec = std::chrono::duration_cast<std::chrono::milliseconds>(ALLendTimer - ALLstartTimer);
            std::cout << "System ALL t_msec = " << ALL_t_msec.count() << std::endl;
            //waitKey(0);
        }
    }
}