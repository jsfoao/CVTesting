#include <iostream>
#include "core.h"

using namespace cv;
using namespace cvext;

int main()
{
    Mat imgRgb = imread("content\\1.jpg");
    imshow("RGB", imgRgb);

    Mat imgGrey = RgbToGrey(imgRgb);
    imshow("Grey", imgGrey);

    Mat imgBin = GreyToBinary(imgGrey);
    imshow("Bin", imgBin);

    Mat imgAverage = Average(imgGrey);
    imshow("3*3 Average", imgAverage);

    imshow("Edge", Edge(imgAverage, 20));

    waitKey();
}