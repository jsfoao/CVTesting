#include <iostream>
#include "core.h"

using namespace cv;
using namespace cvext;

int main()
{
    tesseract::TessBaseAPI* tess = new tesseract::TessBaseAPI();
    if (tess->Init("tesseract-ocr\\tessdata", "eng"))
    {
        std::cout << "Could not initialize Tesseract!" << std::endl;
        exit(1);
    }

    Mat img = imread("content\\therock.jpg");
    imshow("Random", img);

    //Mat img = imread("content\\1.jpg");
    //Mat imgGrey = RgbToGrey(img);
    //Mat imgAverage = Average(imgGrey, 5);
    //Mat imgBin = GreyToBinary(imgAverage);
    //Mat imgBinAverage = Average(imgBin, 3);
    //Mat imgEdge = Edge(imgBinAverage, 20);

    //Mat imgResult = imgEdge;

    //tess->SetImage(imgResult.data, imgResult.cols, imgResult.rows, imgResult.channels(), imgResult.step1());

    //char* outText;
    //outText = tess->GetUTF8Text();

    //imshow("RGB", img);
    //imshow("Processed", imgBin);
    //std::cout << "-------------" << std::endl;
    //std::cout << outText << std::endl;
    //std::cout << "-------------" << std::endl;

    waitKey();

    tess->End();
    //delete tess;
    //delete[] outText;

    //Mat imgRgb = imread("content\\1.jpg");
    //imshow("RGB", imgRgb);

    //Mat imgGrey = RgbToGrey(imgRgb);
    //imshow("Grey", imgGrey);

    //Mat imgBin = GreyToBinary(imgGrey);
    //imshow("Bin", imgBin);

    //Mat imgAverage = Average(imgGrey);
    //imshow("3*3 Average", imgAverage);

    //imshow("Edge", Edge(imgAverage, 20));
}