#include <iostream>
#include "core.h"

using namespace cv;
using namespace extensions;

int main()
{
    Mat imgRgb = imread("content\\1.jpg");
    Mat imgGrey = RgbToGrey(imgRgb);
    Mat imgBinary = GreyToBinary(imgGrey);

    imshow("Grey", imgGrey);
    //imshow("Binary", imgBinary);
    //imshow("Inverted Grey", Invert(imgGrey));
    //imshow("Inverted Binary", Invert(imgBinary));
    //imshow("Stepped Grey", Step(imgGrey, 200));
    //imshow("Averaged", Average(imgGrey, 3));
    imshow("BoxBlur", BoxBlur(imgGrey, 1));

    std::cout << imgBinary.cols << std::endl;
    waitKey();
}