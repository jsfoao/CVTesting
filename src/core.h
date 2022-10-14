#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

namespace cvext
{
    // CV Pre-processing pipeline
    // RGB
    // Grey
    // Blur: Remove noise mostly from smaller objects (Salt and pepper noise)
    // Binary

    Mat RgbToGrey(Mat rgb)
    {
        // CV_8UC1 creates matrix with 1 value for each pixel (for greyscale and binary)
        // CV_8UC3 creates matrix with 3 values for each pixel (for rgb)
        Mat output = Mat::zeros(rgb.size(), CV_8UC1);

        for (int i = 0; i < rgb.rows; i++)
        {
            for (int j = 0; j < rgb.cols * 3; j += 3)
            {
                output.at<uchar>(i, j / 3) = 
                    (rgb.at<uchar>(i, j) 
                    + rgb.at<uchar>(i, j + 1) 
                    + rgb.at<uchar>(i, j + 2)) / 3;
            }
        }

        return output;
    }

    /*
    index = 0: Blue
    index = 1: Green
    index = 2: Red
    */
    Mat RgbToIndex(Mat rgb, int index)
    {
        Mat output = Mat::zeros(rgb.size(), CV_8UC3);

        for (int i = 0; i < rgb.rows; i++)
        {
            for (int j = 0; j < rgb.cols * 3; j += 3)
            {
                output.at<uchar>(i, j + index) = rgb.at<uchar>(i, j);
            }
        }

        return output;
    }

    Mat RgbToBinary(Mat rgb, int threshold = 128)
    {
        Mat output = Mat::zeros(rgb.size(), CV_8UC1);

        for (int i = 0; i < rgb.rows; i++)
        {
            for (int j = 0; j < rgb.cols * 3; j += 3)
            {
                int average = (rgb.at<uchar>(i, j) + rgb.at<uchar>(i, j + 1) + rgb.at<uchar>(i, j + 2)) / 3;
                if (average >= threshold)
                    output.at<uchar>(i, j / 3) = 255;
            }
        }

        return output;
    }

    Mat GreyToBinary(Mat grey, float threshold = 128.f)
    {
        Mat output = Mat::zeros(grey.size(), CV_8UC1);

        for (int i = 0; i < grey.rows; i++)
        {
            for (int j = 0; j < grey.cols; j++)
            {
                if (grey.at<uchar>(i, j) >= threshold)
                    output.at<uchar>(i, j) = 255.f;
            }
        }

        return output;
    }

    Mat Invert(Mat img)
    {
        Mat output = Mat::zeros(img.size(), CV_8UC1);

        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                output.at<uchar>(i, j) = 255.f - img.at<uchar>(i, j);
            }
        }

        return output;
    }

    /*
    th1 exclusive, th2 inclusive
    */
    Mat Step(Mat img, int th1, int th2 = 255)
    {
        Mat output = Mat::zeros(img.size(), CV_8UC1);

        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                uchar val = img.at<uchar>(i, j);
                if (val > th1 && val <= th2)
                    output.at<uchar>(i, j) = 255;
            }
        }

        return output;
    }

    Mat Average(Mat img, int radius = 3)
    {
        Mat output = Mat::zeros(img.size(), CV_8UC1);

        // ensures valid sizes
        int size = (radius * 2) + 1;

        int range = (size - 1) / 2;
        int pixels = pow((2 * range) + 1, 2);

        for (int i = range; i < img.rows - range; i++)
        {
            for (int j = range; j < img.cols - range; j++)
            {
                int sum = 0;
                for (int ii = -range; ii <= range; ii++)
                {
                    for (int jj = -range; jj <= range; jj++)
                    {
                        int value = img.at<uchar>(i + ii, j + jj);
                        sum += value;
                    }
                }
                int average = sum / pixels;
                output.at<uchar>(i, j) = average;
            }
        }
        return output;
    }

    Mat Max(Mat img, int radius = 3)
    {
        Mat output = Mat::zeros(img.size(), CV_8UC1);

        // ensures valid sizes
        int size = (radius * 2) + 1;

        int range = (size - 1) / 2;
        int pixels = pow((2 * range) + 1, 2);

        for (int i = range; i < img.rows - range; i++)
        {
            for (int j = range; j < img.cols - range; j++)
            {
                int max = 0;
                for (int ii = -range; ii <= range; ii++)
                {
                    for (int jj = -range; jj <= range; jj++)
                    {
                        int value = img.at<uchar>(i + ii, j + jj);
                        if (value > max)
                            max = value;
                    }
                }
                output.at<uchar>(i, j) = max;
            }
        }
        return output;
    }

    Mat Min(Mat img, int radius = 3)
    {
        Mat output = Mat::zeros(img.size(), CV_8UC1);

        // ensures valid sizes
        int size = (radius * 2) + 1;

        int range = (size - 1) / 2;
        int pixels = pow((2 * range) + 1, 2);

        for (int i = range; i < img.rows - range; i++)
        {
            for (int j = range; j < img.cols - range; j++)
            {
                int min = 255;
                for (int ii = -range; ii <= range; ii++)
                {
                    for (int jj = -range; jj <= range; jj++)
                    {
                        int value = img.at<uchar>(i + ii, j + jj);
                        if (value < min)
                            min = value;
                    }
                }
                output.at<uchar>(i, j) = min;
            }
        }
        return output;
    }

    Mat Edge(Mat img, int th, int size = 3)
    {
        Mat output = Mat::zeros(img.size(), CV_8UC1);

        int range = (size - 1) / 2;
        for (int i = range; i < img.rows - range; i++)
        {
            for (int j = 1; j < img.cols - 1; j++)
            {
                int sumL = 0;
                int sumR = 0;
                int count = 0;
                for (int ii = -range; ii <= range; ii++)
                {
                    sumL += img.at<uchar>(i + ii, j - 1);
                    sumR += img.at<uchar>(i + ii, j + 1);
                    count++;
                }

                int avgL = sumL / count;
                int avgR = sumR / count;
                if (abs(avgL - avgR) > th)
                    output.at<uchar>(i, j) = 255;
            }
        }
        return output;
    }
};