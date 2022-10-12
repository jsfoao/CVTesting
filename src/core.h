#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

namespace extensions
{
    Mat RGB2Grey(Mat RGB)
    {
        Mat grey = Mat::zeros(RGB.size(), CV_8UC1);

        for (int i = 0; i < RGB.rows; i++)
        {
            for (int j = 0; j < RGB.cols * 3; j += 3)
            {
                grey.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
            }
        }

        return grey;
    }

    Mat RGB2R(Mat RGB)
    {
        Mat red = Mat::zeros(RGB.size(), CV_8UC1);

        for (int i = 0; i < RGB.rows; i++)
        {
            for (int j = 0; j < RGB.cols * 3; j += 3)
            {
                red.at<uchar>(i, j / 3) = RGB.at<uchar>(i, j);
            }
        }

        return red;
    }
};