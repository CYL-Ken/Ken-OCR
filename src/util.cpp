#include <iostream>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>

bool comp(cv::Point2f a, cv::Point2f b)
{
    if (a.y != b.y)
        return a.y < b.y;
    return a.x <= b.x;
}

cv::Mat find_card(cv::Mat input)
{
    std::cout << " === Find Card Start ===" << std::endl;
    cv::Mat result;
    cv::Mat source;
    input.copyTo(source);

    std::cout << " - Gray" << std::endl;
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    // cv::imshow("Gray", gray);

    std::cout << " - Canny" << std::endl;
    cv::Mat processing;
    cv::Canny(gray, processing, 150, 300, 3, true);
    // cv::imshow("Canny", processing);

    std::cout << " - Find Contours" << std::endl;
    cv::RotatedRect resultRect;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(processing, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    if (contours.size() != 0)
    {
        std::cout << "   Contours: " << contours.size() << std::endl;
        double max = -1;
        unsigned int max_index = 0;
        for (unsigned int i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > max)
            {
                max = area;
                max_index = i;
            }
        }

        resultRect = cv::minAreaRect(contours[max_index]);
        cv::Point2f pt[4];
        resultRect.points(pt);

        // cv::line(input, pt[0], pt[1], cv::Scalar(255, 0, 0), 2, 8);
        // cv::line(input, pt[1], pt[2], cv::Scalar(255, 0, 0), 2, 8);
        // cv::line(input, pt[2], pt[3], cv::Scalar(255, 0, 0), 2, 8);
        // cv::line(input, pt[3], pt[0], cv::Scalar(255, 0, 0), 2, 8);
        // cv::imshow("Line", input);
    }

    cv::Point2f sp[4];
    resultRect.points(sp);
    std::sort(sp, sp + 4, comp);
    int bw = int(resultRect.size.width);
    int bh = int(resultRect.size.height);
    if (bw < bh)
    {
        int tmp = bw;
        bw = bh;
        bh = tmp;
    }
    cv::Mat M;
    if (sp[0].x > sp[1].x)
    {
        cv::Point2f dp[] = {cv::Point2f(bw - 1, 0), cv::Point2f(0, 0), cv::Point2f(bw - 1, bh - 1), cv::Point2f(0, bh - 1)};
        M = getPerspectiveTransform(sp, dp);
    }
    else
    {
        cv::Point2f dp[] = {cv::Point2f(0, 0), cv::Point2f(bw - 1, 0), cv::Point2f(0, bh - 1), cv::Point2f(bw - 1, bh - 1)};
        M = getPerspectiveTransform(sp, dp);
    }
    warpPerspective(source, result, M, cv::Size(bw, bh));

    std::cout << " === Find Card Finished ===" << std::endl;
    return result;
}

cv::Mat find_target(cv::Mat input, std::string target, cv::Point fov_top, cv::Point fov_down)
{
    cv::Mat result;
    cv::Mat source;
    cv::imshow("Input", input);

    std::cout << " === Find " << target << " on Card === " << std::endl;

    cv::Rect fov_roi(fov_top, fov_down);

    cv::Mat fov;
    input(fov_roi).copyTo(fov);

    cv::imshow("FOV", fov);
    cv::waitKey(0);
    std::cout << " - Gray" << std::endl;
    cv::Mat gray;
    cv::cvtColor(fov, gray, cv::COLOR_BGR2GRAY);

    std::cout << " - Thershold" << std::endl;
    cv::Mat dst;
    cv::threshold(gray, dst, 210, 255, cv::THRESH_BINARY);

    result = dst.clone();

    std::cout << " - Erode" << std::endl;
    cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 10));
    cv::erode(dst, dst, erodeElement);

    std::cout << " - Find Contours" << std::endl;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dst, contours, hierarchy,
                     cv::RETR_CCOMP,
                     cv::CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0));

    if (contours.size() != 0)
    {
        std::cout << " - Got " << contours.size() << " Contours" << std::endl;
        double max = -1;
        unsigned int max_index = 0;
        for (unsigned int i = 0; i < contours.size(); i++)
        {
            if (hierarchy[i][3] == -1)
                continue;
            double area = cv::contourArea(contours[i]);
            if (area > max)
            {
                max = area;
                max_index = i;
            }
        }

        cv::Rect rect = cv::boundingRect(contours[max_index]);
        cv::Point top(rect.x, rect.y);
        cv::Point down(rect.br().x, rect.br().y);

        if (abs(rect.br().x - rect.x) < 20 || abs(rect.br().y - rect.y) < 20)
        {
            cv::Mat NULLMAT;
            return NULLMAT;
        }
        cv::Rect roi(top, down);
        // source(roi).copyTo(result);
        std::cout << " === Find " << target << " Success === " << std::endl;
        return result(roi);
    }
    else
    {
        std::cout << " - Cannot Find Target." << std::endl;
        cv::Mat NULLMAT;
        return NULLMAT;
    }
}