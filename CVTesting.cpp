#include <iostream>
#include "core.h"

using namespace cv;
using namespace cvext;

int main()
{
    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
    if (api->Init("tesseract-ocr\\tessdata", "eng"))
    {
        std::cout << "Could not initialize Tesseract!" << std::endl;
        exit(1);
    }

    Mat img = imread("content\\1.jpg");
    imshow("RGB", img);

    Mat imgGrey = RgbToGrey(img);
    imshow("Grey", imgGrey);

    Mat imgAvg = Average(imgGrey, 1);
    imshow("Average", imgAvg);

    Mat imgEdge = VerticalEdge(imgAvg, 50);
    imshow("Edge", imgEdge);

    Mat imgErosion = Erosion(imgEdge);
    imshow("Erosion", imgErosion);

    Mat imgDilated = Dilation(imgErosion, 17);
    imshow("Dilation 3*3", imgDilated);

    Mat imgID = Segmentation(imgDilated);
    Mat imgIDGrey = IDToGrey(imgID);
    imshow("IDs", imgIDGrey);


    api->SetImage(imgGrey.data, imgGrey.cols, imgGrey.rows, imgGrey.channels(), imgGrey.step1());

    char* outText;
    outText = api->GetUTF8Text();

    std::cout << "-------------" << std::endl;
    std::cout << outText << std::endl;
    std::cout << "-------------" << std::endl;

    waitKey();

    api->End();
    delete api;
    delete[] outText;
}

    //Mat imgAverage = Average(imgGrey, 5);
    //Mat imgBin = GreyToBinary(imgAverage);
    //Mat imgBinAverage = Average(imgBin, 3);
    //Mat imgEdge = Edge(imgBinAverage, 20);
    
    //Mat imgResult = imgEdge;
    
    //Mat imgRgb = imread("content\\1.jpg");
    //imshow("RGB", imgRgb);

    //Mat imgGrey = RgbToGrey(imgRgb);
    //imshow("Grey", imgGrey);

    //Mat imgBin = GreyToBinary(imgGrey);
    //imshow("Bin", imgBin);

    //Mat imgAverage = Average(imgGrey);
    //imshow("3*3 Average", imgAverage);

    //imshow("Edge", Edge(imgAverage, 20));