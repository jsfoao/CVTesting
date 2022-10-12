#include <iostream>
#include "core.h"

using namespace cv;
using namespace extensions;

int main()
{
    Mat imgRgb = imread("content\\1.jpg");

    // CV_8UC1 creates matrix with 1 value for each pixel (for greyscale and binary)
    // CV_8UC3 creates matrix with 3 values for each pixel (for rgb)
    Mat imgGrey = RGB2R(imgRgb);
    
    imshow("Grey Image", imgGrey);
    waitKey();
}