#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "vcpkg/tesseract/baseapi.h"
#include "vcpkg/leptonica/allheaders.h"
#include <random>
#include <vector>

using namespace std;
using namespace cv;

namespace cvext
{
    // CV Pre-processing pipeline
    // RGB
    // Grey
    // Blur: Remove noise mostly from smaller objects (Salt and pepper noise)
    // Binary
    struct CharBox
    {
        char Char;
        int x;
        int y;

        CharBox(char character, int x, int y)
        {
            Char = character;
            this->x = x;
            this->y = y;
        }
    };

    float FillRatio(Mat img);

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

    Mat Invert(Mat imgBin)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        for (int i = 0; i < imgBin.rows; i++)
        {
            for (int j = 0; j < imgBin.cols; j++)
            {
                output.at<uchar>(i, j) = 255 - imgBin.at<uchar>(i, j);
            }
        }

        return output;
    }

    /*
    th1 exclusive, th2 inclusive
    */
    Mat Step(Mat imgBin, int th1, int th2 = 255)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        for (int i = 0; i < imgBin.rows; i++)
        {
            for (int j = 0; j < imgBin.cols; j++)
            {
                uchar val = imgBin.at<uchar>(i, j);
                if (val > th1 && val <= th2)
                    output.at<uchar>(i, j) = 255;
            }
        }

        return output;
    }

    Mat Average(Mat imgBin, int radius = 3)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        // ensures valid sizes
        int size = (radius * 2) + 1;

        int range = (size - 1) / 2;
        int pixels = pow((2 * range) + 1, 2);

        for (int i = range; i < imgBin.rows - range; i++)
        {
            for (int j = range; j < imgBin.cols - range; j++)
            {
                int sum = 0;
                for (int ii = -range; ii <= range; ii++)
                {
                    for (int jj = -range; jj <= range; jj++)
                    {
                        int value = imgBin.at<uchar>(i + ii, j + jj);
                        sum += value;
                    }
                }
                int average = sum / pixels;
                output.at<uchar>(i, j) = average;
            }
        }
        return output;
    }

    Mat Max(Mat imgBin, int radius = 3)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        // ensures valid sizes
        int size = (radius * 2) + 1;

        int range = (size - 1) / 2;
        int pixels = pow((2 * range) + 1, 2);

        for (int i = range; i < imgBin.rows - range; i++)
        {
            for (int j = range; j < imgBin.cols - range; j++)
            {
                int max = 0;
                for (int ii = -range; ii <= range; ii++)
                {
                    for (int jj = -range; jj <= range; jj++)
                    {
                        int value = imgBin.at<uchar>(i + ii, j + jj);
                        if (value > max)
                            max = value;
                    }
                }
                output.at<uchar>(i, j) = max;
            }
        }
        return output;
    }

    Mat Min(Mat imgBin, int radius = 3)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        // ensures valid sizes
        int size = (radius * 2) + 1;

        int range = (size - 1) / 2;
        int pixels = pow((2 * range) + 1, 2);

        for (int i = range; i < imgBin.rows - range; i++)
        {
            for (int j = range; j < imgBin.cols - range; j++)
            {
                int min = 255;
                for (int ii = -range; ii <= range; ii++)
                {
                    for (int jj = -range; jj <= range; jj++)
                    {
                        int value = imgBin.at<uchar>(i + ii, j + jj);
                        if (value < min)
                            min = value;
                    }
                }
                output.at<uchar>(i, j) = min;
            }
        }
        return output;
    }

    Mat Edge(Mat imgBin, int th, int size = 3)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        int range = (size - 1) / 2;
        for (int i = range; i < imgBin.rows - range; i++)
        {
            for (int j = 1; j < imgBin.cols - 1; j++)
            {
                int sumL = 0;
                int sumR = 0;
                int count = 0;
                for (int ii = -range; ii <= range; ii++)
                {
                    sumL += imgBin.at<uchar>(i + ii, j - 1);
                    sumR += imgBin.at<uchar>(i + ii, j + 1);
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

    Mat Dilation(Mat imgBin, int range = 1)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        for (int i = range; i < imgBin.rows - range; i++)
        {
            for (int j = range; j < imgBin.cols - range; j++)
            {
                // skip if pixel is already white
                if (imgBin.at<uchar>(i, j) == 255)
                {
                    output.at<uchar>(i, j) = 255;
                    continue;
                }

                for (int ii = -range; ii <= range; ii++)
                {
                    for (int jj = -range; jj <= range; jj++)
                    {
                        int value = imgBin.at<uchar>(i + ii, j + jj);

                        // make pixel white if it has white neighbours
                        if (value == 255)
                        {
                            output.at<uchar>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
        }
        return output;
    }

    Mat HorizontalDilation(Mat imgBin, int range = 1)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        for (int i = range; i < imgBin.rows - range; i++)
        {
            for (int j = range; j < imgBin.cols - range; j++)
            {
                // skip if pixel is already white
                if (imgBin.at<uchar>(i, j) == 255)
                {
                    output.at<uchar>(i, j) = 255;
                    continue;
                }

                for (int jj = -range; jj <= range; jj++)
                {
                    int value = imgBin.at<uchar>(i, j + jj);

                    // make pixel white if it has white neighbours
                    if (value == 255)
                    {
                        output.at<uchar>(i, j) = 255;
                        break;
                    }
                }
            }
        }
        return output;
    }

    Mat EqHist(Mat img)
    {
        Mat EQImg = Mat::zeros(img.size(), CV_8UC1);
        // count
        int count[256] = { 0 };
        for (int i = 0; i < img.rows; i++)
            for (int j = 0; j < img.cols; j++)
                count[img.at<uchar>(i, j)]++;


        // prob
        float prob[256] = { 0.0 };
        for (int i = 0; i < 256; i++)
            prob[i] = (float)count[i] / (float)(img.rows * img.cols);

        // accprob
        float accprob[256] = { 0.0 };
        accprob[0] = prob[0];
        for (int i = 1; i < 256; i++)
            accprob[i] = prob[i] + accprob[i - 1];
        // new = 255 * accprob 
        int newvalue[256] = { 0 };
        for (int i = 0; i < 256; i++)
            newvalue[i] = 255 * accprob[i];

        for (int i = 0; i < img.rows; i++)
            for (int j = 0; j < img.cols; j++)
                EQImg.at<uchar>(i, j) = newvalue[img.at<uchar>(i, j)];

        return EQImg;
    }

    int OTSU(Mat img)
    {
        int count[256] = { 0 };
        for (int i = 0; i < img.rows; i++)
            for (int j = 0; j < img.cols; j++)
                count[img.at<uchar>(i, j)]++;


        // prob
        float prob[256] = { 0.0 };
        for (int i = 0; i < 256; i++)
            prob[i] = (float)count[i] / (float)(img.rows * img.cols);

        // accprob
        float theta[256] = { 0.0 };
        theta[0] = prob[0];
        for (int i = 1; i < 256; i++)
            theta[i] = prob[i] + theta[i - 1];

        float meu[256] = { 0.0 };
        for (int i = 1; i < 256; i++)
            meu[i] = i * prob[i] + meu[i - 1];

        float sigma[256] = { 0.0 };
        for (int i = 0; i < 256; i++)
            sigma[i] = pow(meu[255] * theta[i] - meu[i], 2) / (theta[i] * (1 - theta[i]));

        int index = 0;
        float maxVal = 0;
        for (int i = 0; i < 256; i++)
        {
            if (sigma[i] > maxVal)
            {
                maxVal = sigma[i];
                index = i;
            }
        }

        return index + 30;

    }

    int StepFillTh(Mat img, int min = 160, int max = 240)
    {
        float fill = FillRatio(img);
        int output = (fill * 255) + 40;

        if (output > 240)
            output = 240;

        if (output < 160)
            output = 160;

        return output;
    }

    Mat Erosion(Mat imgBin, int range = 1)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        for (int i = range; i < imgBin.rows - range; i++)
        {
            for (int j = range; j < imgBin.cols - range; j++)
            {
                // skip if pixel is already black
                if (imgBin.at<uchar>(i, j) == 0)
                    continue;

                output.at<uchar>(i, j) = 255;
                for (int ii = -range; ii <= range; ii++)
                {
                    for (int jj = -range; jj <= range; jj++)
                    {
                        int value = imgBin.at<uchar>(i + ii, j + jj);

                        // make pixel black if it has black neighbours
                        if (value == 0)
                        {
                            output.at<uchar>(i, j) = 0;
                            break;
                        }
                    }
                }
            }
        }
        return output;
    }

    Mat ColorErosion(Mat* dilated)
    {
        Mat DilatedImgCpy;
        DilatedImgCpy = dilated->clone();

        std::vector<std::vector<Point>> contours1;
        vector<Vec4i> hierachy1;

        findContours(*dilated, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
        Mat dst = Mat::zeros(dilated->size(), CV_8UC3);

        if (!contours1.empty())
        {
            for (int i = 0; i < contours1.size(); i++)
            {
                Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
                drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
            }
        }

        Rect rect;
        Scalar black = CV_RGB(0, 0, 0);

        for (int i = 0; i < contours1.size(); i++)
        {
            rect = boundingRect(contours1[i]);
            float ratio = float(rect.width) / float(rect.height);

            if (rect.width < 40 || rect.height > 100 || rect.width > 150 || rect.x < 0.1 * dilated->cols || rect.x > 0.9 * dilated->cols ||
                rect.y < 0.1 * dilated->rows || rect.y > 0.9 * dilated->rows || ratio < 1.5)
            {
                drawContours(DilatedImgCpy, contours1, i, black, -1, 8, hierachy1);
            }
        }

        return DilatedImgCpy;
    }

    #define NULL 0

    struct Pixel
    {
        int Value;
        int ID;

        Pixel(int value, int id)
        {
            Value = value;
            ID = id;
        }
    };

    Mat IDToGrey(Mat imgBin)
    {
        Mat output = Mat::zeros(imgBin.size(), CV_8UC1);

        for (int i = 0; i < imgBin.rows; i++)
        {
            for (int j = 0; j < imgBin.cols; j++)
            {
                if (imgBin.at<uchar>(i, j) != NULL)
                {
                    output.at<uchar>(i, j) = imgBin.at<uchar>(i, j) * 25;
                }
            }
        }

        return output;
    }

    Mat RandomGrey(Size size, int min = 0, int max = 255)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(min, max);

        Mat output = Mat::zeros(size, CV_8UC1);

        for (int i = 0; i < output.rows; i++)
        {
            for (int j = 0; j < output.cols; j++)
            {
                output.at<uchar>(i, j) = uni(rng);
            }
        }

        return output;
    }

    Mat RandomRGB(Size size, int min = 0, int max = 255)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(min, max);

        Mat output = Mat::zeros(size, CV_8UC3);

        for (int i = 0; i < output.rows; i++)
        {
            for (int j = 0; j < output.cols * 3; j++)
            {
                output.at<uchar>(i, j) = uni(rng);
            }
        }

        return output;
    }

    float FillRatio(Mat imgBin)
    {
        int num = 0;
        int total = imgBin.rows * imgBin.cols;

        for (int i = 0; i < imgBin.rows; i++)
        {
            for (int j = 0; j < imgBin.cols; j++)
            {
                int value = imgBin.at<uchar>(i, j);
                if (value == 255)
                {
                    num++;
                }
            }
        }
        return (float)num / (float)total;
    }

    Mat CopyWithBorder(Mat img, int border)
    {
        Size size(img.cols + (border * 2), img.rows + (border * 2));
        Mat output = Mat::zeros(size, CV_8UC1);

        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                int value = img.at<uchar>(i, j);
                output.at<uchar>(i + border, j + border) = value;
            }
        }

        return output;
    }

    bool IsAdded(int index, vector<int> list)
    {
        for (int i = 0; i < list.size(); i++)
        {
            if (list[i] == index)
            {
                return true;
            }
        }
        return false;
    }

    char* SortedCharBox(vector<CharBox> charBox)
    {
        char* output = new char[charBox.size()]{};
        vector<CharBox> charBoxCpy = charBox;
        int min = 9999;
        int k = 0;
        int minIndex = -1;
        char minChar;
        
        while (k != charBox.size())
        {
            for (int i = 0; i < charBoxCpy.size(); i++)
            {

                int value = charBoxCpy[i].x;
                if (value < min)
                {
                    min = value;
                    minIndex = i;
                    minChar = charBoxCpy[minIndex].Char;
                }
            }
            output[k] = minChar;
            charBoxCpy.erase(charBoxCpy.begin() + minIndex);
            minIndex = -1;
            k++;
            min = 9999;
        }
        return output;
    }
};