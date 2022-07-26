#include "App.h"

App::App()
{	
    std::cout << "init OCR engine: \n";
    
    auto startTimer = std::chrono::steady_clock::now();
    
    api_eng_id = new tesseract::TessBaseAPI();
    if (api_eng_id->Init(NULL, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }
    
    api_chi_name = new tesseract::TessBaseAPI();
    if (api_chi_name->Init(NULL, "chi_tra")) {
        fprintf(stderr, "Could not initialize Chinese tesseract.\n");
        exit(1);
    }
    
    auto endTimer = std::chrono::steady_clock::now();
    std::chrono::milliseconds t_msec = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer - startTimer);
   
    std::cout << "init t_msec = " << t_msec.count() << std::endl;
    
}

App::~App()
{
    api_eng_id->End();
    delete api_eng_id;
    
    api_chi_name->End();
    delete api_chi_name;
    //delete ocrEngine;
}

bool App::process()
{
    std::cout << "start processing\n";
    
    // read image
    // bg image
    const char *bgFname = "../data/bg.png";
    cv::Mat bg_g = cv::imread(bgFname, 0);
    if (bg_g.empty())
    {
        cout << " background File is empty : check file path\n";
        exit(-1);
    }

    cv::Mat bgs_g;
    float scale = 0.5;
    cv::resize(bg_g, bgs_g, 
        cv::Size(static_cast<int>(scale * bg_g.cols), 
                static_cast<int>(scale * bg_g.rows)));

    // list of filePaths
    const char *imlistFname = "../data/imlist.txt";
    vector<string> imlist = getList(imlistFname);

    for (size_t i = 0; i < imlist.size(); i++) 
    {
        auto startTimer = std::chrono::steady_clock::now();
        
        //// find card location
        string imfname = imlist[i];
        cv::Mat I_rgb = cv::imread(imfname, cv::IMREAD_UNCHANGED);

        if (I_rgb.empty())
        {
            cout << " File: " << imfname << " is empty : check file path\n";
            continue;
        }

        cv::Mat I_g, Is_g, Is_rgb;
        cv::cvtColor(I_rgb, I_g, cv::COLOR_BGR2GRAY);
        
        cv::resize(I_g, Is_g, 
            cv::Size(static_cast<int>(scale * I_rgb.cols), static_cast<int>(scale * I_rgb.rows)));
        cv::resize(I_rgb, Is_rgb, 
            cv::Size(static_cast<int>(scale * I_rgb.cols), static_cast<int>(scale * I_rgb.rows)));

        // calculate absolute difference of current frame and background
        cv::Mat dframe;
        cv::absdiff(bgs_g, Is_g, dframe);
        //cv::imshow("dframe",dframe);
        cv::threshold(dframe, dframe, 36, 255, cv::THRESH_BINARY);
        
        cv::imshow("dframe",dframe);
        // Find contours
        vector<vector<cv::Point>> contours;
        cv::findContours(dframe, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

        // sort contours
        std::sort(contours.begin(), contours.end(), 
            [](std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
                    double i = fabs(cv::contourArea(cv::Mat(contour1)));
                    double j = fabs(cv::contourArea(cv::Mat(contour2)));
                    return ( i < j );
        });

        // grab contours
        std::vector<cv::Point> biggestContour = contours[contours.size()-1];
        std::vector<cv::Point> smallestContour = contours[0];

        cv::RotatedRect rRect = cv::minAreaRect(biggestContour);
        
        // warp card
        cv::Point2f sp[4], new_sp[4];
        rRect.points(sp);
        double d1 = cv::norm(sp[0]-sp[1]);
        double d2 = cv::norm(sp[1]-sp[2]);
        if (d1 < d2) {
            new_sp[0] = sp[1];
            new_sp[1] = sp[2];
            new_sp[2] = sp[3];
            new_sp[3] = sp[0];
        }
        else {
            rRect.points(new_sp);
        }

        int bw = int(rRect.size.width);
        int bh = int(rRect.size.height);

        if (bw < bh)
        {
            int tmp = bw;
            bw = bh;
            bh = tmp;
        }

        for (int i = 0; i < 4; i++)
            new_sp[i] = new_sp[i] * 2;
        bw *= 2;
        bh *= 2;

        cv::Point2f dp[] = {cv::Point2f(0, 0), cv::Point2f(bw-1, 0), 
                            cv::Point2f(bw-1, bh-1), cv::Point2f(0, bh-1)};

        cv::Mat M = cv::getPerspectiveTransform(new_sp, dp);

        cv::Mat I_rgb_warped;
        cv::warpPerspective(I_rgb, I_rgb_warped, M, cv::Size(bw, bh));

        /*
        sp[0] = new_sp[2];
        sp[1] = new_sp[3];
        sp[2] = new_sp[0];
        sp[3] = new_sp[1];
        */
        cv::imshow("I_rgb",I_rgb);
        cv::imshow("I_rgb_warped",I_rgb_warped);
        // recognize ID:
        // get ID location
        int id_left =  floor((2.65 / 8.6) * bw);
        int id_right =  floor((5.3 / 8.6) * bw);
        int id_top =  floor((3.5 / 5.5) * bh);
        int id_bottom =  floor((4.0 / 5.5) * bh);
        cv::Mat id_Img = I_rgb_warped(cv::Range(id_top, id_bottom), cv::Range(id_left, id_right));

        overSegID(id_Img);

        // setup Name ocrEngine
        // get Name location
        int name_left =  floor((2.4 / 8.6) * bw);
        int name_right =  floor((5.0 / 8.6) * bw);
        int name_top =  floor((2.1 / 5.5) * bh);
        int name_bottom =  floor((3.2 / 5.5) * bh);
        cv::Mat name_Img = I_rgb_warped(cv::Range(name_top, name_bottom), cv::Range(name_left, name_right));

        overSegName(name_Img);
        
        auto endTimer = std::chrono::steady_clock::now();
        std::chrono::milliseconds t_msec = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer - startTimer);
    }
}

vector<string> App::getList(const char* fname) 
{   
    vector<string> names(0); 
    string str;
    ifstream file(fname);

    if(!file.is_open()){
        cerr << "Failed opening file! " << fname << endl;
        return vector<string>();
    }
    while(!file.eof()){
        getline(file,str); 
        if(str.length() < 2) 
            continue;
        names.push_back(str);
    }
    file.close(); return names;
}

bool App::overSegID(cv::Mat id_img)
{   
    cv::Mat id_img_g, b_img;

    cv::cvtColor(id_img, id_img_g, cv::COLOR_BGR2GRAY);

    cv::threshold(id_img_g, b_img, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Calculate the histogram for vertical stripes
    int columnCount, rowCount;
    std::vector<int> colHist, rowHist;

    for (int col = 0; col < id_img.cols; col++)
    {
        columnCount = 0;

        for (int row = 0; row < id_img.rows; row++)
        {
            if (b_img.at<uchar>(row, col) == 0)
                columnCount++;
        }

        colHist.push_back(columnCount);
    }
    
    for (int row = 0; row < id_img.rows; row++)
    {
        rowCount = 0;

        for (int col = 0; col < id_img.cols; col++)
        {
            if (b_img.at<uchar>(row, col) == 0)
                rowCount++;
        }

        rowHist.push_back(rowCount);
    }

    std::vector<std::pair<int, int> > zeros_col = zero_runs(colHist, 0);
    std::vector<std::pair<int, int> > zeros_row = zero_runs(rowHist, 0);
    
    cv::imshow("id_img",id_img);
    
    cv::Mat id_letter_img = id_img(cv::Range(zeros_row[0].second-10, zeros_row[zeros_row.size()-1].first+10), cv::Range(zeros_col[0].second-5, zeros_col[1].first+5));
    cv::imshow("id_letter_img",id_letter_img);
    api_eng_id->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    api_eng_id->SetImage((uchar*)id_letter_img.data, id_letter_img.size().width, id_letter_img.size().height, id_letter_img.channels(), id_letter_img.step1());
    api_eng_id->Recognize(0);
    char* outText_letter = api_eng_id->GetUTF8Text();
    std::cout << "letter String = " << std::string(outText_letter) << std::endl;

    cv::Mat id_number_img = id_img(cv::Range(zeros_row[0].second-10, zeros_row[zeros_row.size()-1].first+10), cv::Range(zeros_col[1].second-5, zeros_col[zeros_col.size()-1].first+5));
    cv::imshow("id_number_img",id_number_img);
    std::string num_whitelist = "0123456789";
    api_eng_id->SetVariable("tessedit_char_whitelist", "0123456789");
    api_eng_id->SetImage((uchar*)id_number_img.data, id_number_img.size().width, id_number_img.size().height, id_number_img.channels(), id_number_img.step1());
    api_eng_id->Recognize(0);
    char* outText_num = api_eng_id->GetUTF8Text();
    std::cout << "number String = " << std::string(outText_num) << std::endl;
    cv::waitKey(0);

    return true;
}

std::vector<std::pair<int, int> > App::zero_runs(std::vector<int> histo, int type)
{
    vector<pair<int, int>> zeros;
    bool is_counting = false;
    int start = -1, end = -1; 
    for (int i = 0; i < histo.size(); i++)
    {
        if (histo[i] == 0) {
            if (!is_counting) {
                start = i;
                is_counting = true;
            }
        }
        else {
            if (is_counting) {
                end = i;
                is_counting = false;

                int margin;
                if (type == 0)
                    margin = 5;
                else
                    margin = 15;

                if (end - start > margin) {    
                    pair<int, int> local_pair(start, end);
                    zeros.push_back(local_pair);
                }
            }
        }
    }
    pair<int, int> local_pair(start, histo.size()-1);
    zeros.push_back(local_pair);
    return zeros;
}

bool App::overSegName(cv::Mat name_img)
{   
    cv::Mat name_img_g, b_img;

    cv::cvtColor(name_img, name_img_g, cv::COLOR_BGR2GRAY);

    cv::threshold(name_img_g, b_img, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Calculate the histogram for vertical stripes
    int columnCount, rowCount;
    std::vector<int> colHist, rowHist;

    for (int col = 0; col < name_img.cols; col++)
    {
        columnCount = 0;

        for (int row = 0; row < name_img.rows; row++)
        {
            if (b_img.at<uchar>(row, col) == 0)
                columnCount++;
        }

        colHist.push_back(columnCount);
    }
    
    for (int row = 0; row < name_img.rows; row++)
    {
        rowCount = 0;

        for (int col = 0; col < name_img.cols; col++)
        {
            if (b_img.at<uchar>(row, col) == 0)
                rowCount++;
        }

        rowHist.push_back(rowCount);
    }

    std::vector<std::pair<int, int> > zeros_col = zero_runs(colHist, 1);
    std::vector<std::pair<int, int> > zeros_row = zero_runs(rowHist, 1);
    
    for (int i = 1; i < zeros_col.size(); i++)
    {
        int start = zeros_col[i-1].second-5;
        int end = zeros_col[i].first+5;
        cv::Mat name_char_img = name_img(cv::Range(zeros_row[0].second-10, zeros_row[zeros_row.size()-1].first+10), cv::Range(start, end));
        
        api_chi_name->SetImage((uchar*)name_char_img.data, name_char_img.size().width, name_char_img.size().height, name_char_img.channels(), name_char_img.step1());
        api_chi_name->Recognize(0);
        char* outText_name = api_chi_name->GetUTF8Text();
        std::cout << "char = " << std::string(outText_name) << std::endl;
    }
    
    return true;
}
