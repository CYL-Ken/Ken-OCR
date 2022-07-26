#ifndef __CGMHDATA_H__
#define __CGMHDATA_H__

#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <curl/curl.h>
#include <jsoncpp/json/json.h>
using namespace std;



class cgmhDataSender
{

public:
    cgmhDataSender();
    ~cgmhDataSender();

    int getI500Data(string i500Ip, string fileDst);

    int genData(string lastName, string timeStamp, string personId);
    
    int collectCgmhData();
   
    int writeRequiredImg(cv::Mat faceFrame, cv::Mat idCardFrame);
   
    int sendCgmhData(string cgmhUrl);
    
    int sendCgmhDataTEST(string cgmhUrl,string _JsonFile);

private:
    string _maskOn;
    string _temperature;
    string _nameDataJson;
    string _nameFaceImg;
    string _nameIdCardImg;
};
#endif // __CGMHDATA_H__
