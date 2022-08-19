#include <iostream>
#include <string.h>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>

#include "util.h"

// using namespace std;

/**
 * Test Image Processing Function
 */

typedef struct CardInfo
{
    cv::Point id_top;
    cv::Point id_down;
} CardInfo;

CardInfo load_config(CardInfo card_info)
{
    cv::FileStorage config("config.xml", cv::FileStorage::READ);
    config["ID_top_x"] >> card_info.id_top.x;
    config["ID_top_y"] >> card_info.id_top.y;
    config["ID_down_x"] >> card_info.id_down.x;
    config["ID_down_y"] >> card_info.id_down.y;

    return card_info;
}

int main()
{
    std::cout << " === ORC SYSTEM ===" << std::endl;
    std::cout << "[Load OCR Config File]" << std::endl;
    CardInfo card_info;
    try
    {
        card_info = load_config(card_info);
    }
    catch (cv::Exception &e)
    {
        std::cerr << e.what();
        return 0;
    }

    // Test by Read Image
    std::cout << "[Test Image Process Function]" << std::endl;
    std::cout << " - Load Image" << std::endl;
    cv::Mat image = cv::imread("../figure/3.jpg");

    // Find Card
    cv::Mat card = find_card(image);
    cv::imshow("Card", card);

    // Find Name and ID
    cv::Mat id_image = find_target(card, "ID", card_info.id_top, card_info.id_down);
    cv::imshow("id_image", id_image);
    // cv::Point name_top(300, 245);
    // cv::Point name_down(530, 440);
    // cv::Mat name_image = find_target(card, "Name", name_top, name_down);

    // if(id_image.rows != 0){
    //     cv::imshow("ID", id_image);
    // }

    cv::waitKey(0);

    return 0;
}
