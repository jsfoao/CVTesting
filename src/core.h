#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

namespace extensions
{
    Mat RgbToGrey(Mat rgb)
    {
        // CV_8UC1 creates matrix with 1 value for each pixel (for greyscale and binary)
        // CV_8UC3 creates matrix with 3 values for each pixel (for rgb)
        Mat grey = Mat::zeros(rgb.size(), CV_8UC1);

        for (int i = 0; i < rgb.rows; i++)
        {
            for (int j = 0; j < rgb.cols * 3; j += 3)
            {
                grey.at<uchar>(i, j / 3) = 
                    (rgb.at<uchar>(i, j) 
                    + rgb.at<uchar>(i, j + 1) 
                    + rgb.at<uchar>(i, j + 2)) / 3;
            }
        }

        return grey;
    }

    /*
    index = 0: Blue
    index = 1: Green
    index = 2: Red
    */
    Mat RgbToIndex(Mat rgb, int index)
    {
        Mat red = Mat::zeros(rgb.size(), CV_8UC3);

        for (int i = 0; i < rgb.rows; i++)
        {
            for (int j = 0; j < rgb.cols * 3; j += 3)
            {
                red.at<uchar>(i, j + index) = rgb.at<uchar>(i, j);
            }
        }

        return red;
    }

    Mat RgbToBinary(Mat rgb, float threshold = 128.f)
    {
        Mat binary = Mat::zeros(rgb.size(), CV_8UC1);

        for (int i = 0; i < rgb.rows; i++)
        {
            for (int j = 0; j < rgb.cols * 3; j += 3)
            {
                float average = (rgb.at<uchar>(i, j) + rgb.at<uchar>(i, j + 1) + rgb.at<uchar>(i, j + 2)) / 3;
                if (average >= threshold)
                    binary.at<uchar>(i, j / 3) = 255.f;
            }
        }

        return binary;
    }

    Mat GreyToBinary(Mat grey, float threshold = 128.f)
    {
        Mat binary = Mat::zeros(grey.size(), CV_8UC1);

        for (int i = 0; i < grey.rows; i++)
        {
            for (int j = 0; j < grey.cols; j++)
            {
                if (grey.at<uchar>(i, j) >= threshold)
                    binary.at<uchar>(i, j) = 255.f;
            }
        }

        return binary;
    }

    Mat Invert(Mat img)
    {
        Mat inverted = Mat::zeros(img.size(), CV_8UC1);

        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                inverted.at<uchar>(i, j) = 255.f - img.at<uchar>(i, j);
            }
        }

        return inverted;
    }

    /*
    th1 exclusive, th2 inclusive
    */
    Mat Step(Mat img, int th1, int th2 = 255)
    {
        Mat stepped = Mat::zeros(img.size(), CV_8UC1);

        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                uchar val = img.at<uchar>(i, j);
                if (val > th1 && val <= th2)
                    stepped.at<uchar>(i, j) = 255;
            }
        }

        return stepped;
    }

    Mat Average(Mat img, int size)
    {
        Mat averaged = Mat::zeros(img.size(), CV_8UC1);

        for (int i = 0 + size; i < img.rows - size; i++)
        {
            for (int j = 0 + size; j < img.cols - size; j++)
            {
                int sum = 0;
                int neighbourCount = 0;
                for (int x = -size; x < size; x++)
                {
                    for (int y = -size; y < size; y++)
                    {
                        if (x != 0 && y != 0)
                        {
                            sum += img.at<uchar>(i + x, j + y);
                            neighbourCount++;
                        }
                    }
                }
                int average = sum / neighbourCount;
                img.at<uchar>(i, j) = average;
            }
        }

        return img;
    }
};