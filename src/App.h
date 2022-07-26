#ifndef APP_H
#define APP_H

//#include "PreProcess.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <chrono>
#include <future>

using namespace std;

class App
{
public:
    App();
    ~App();
    bool process();

private:
    vector<string> getList(const char* fname);
    bool overSegID(cv::Mat id_img);
    bool overSegName(cv::Mat name_img);
    std::vector<std::pair<int, int> > zero_runs(std::vector<int>, int type);
    
    tesseract::TessBaseAPI *api_chi_name, *api_eng_id;
    
};

#endif // APP_H
