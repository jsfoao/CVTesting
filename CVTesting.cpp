#include <iostream>
#include "core.h"

using namespace cv;
using namespace cvext;
using namespace std;

#define NUM_IMAGES 1

int main()
{
    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
    if (api->Init("tesseract-ocr\\tessdata", "eng"))
    {
        std::cout << "Could not initialize Tesseract!" << std::endl;
        exit(1);
    }

    api->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_CHAR);


    String path = "content\\19.jpg";
    //String path = "content\\" + to_string(i) + ".jpg";
    Mat imgBin = imread(path);
    //Mat imgGreyPre = ;
    Mat imgGrey = RgbToGrey(imgBin);
    Mat imgAvg = Average(imgGrey, 1);
    Mat imgEdge = Edge(imgAvg, 50);
    Mat imgErosion = Erosion(imgEdge, 1);
    Mat imgDilated = Dilation(imgErosion, 15);

    imshow("Grey", imgGrey);
    imshow("Final", imgDilated);

    Mat dilated_cpy = imgDilated.clone();
    Mat plate;
    Rect rect;
    Scalar black = CV_RGB(0, 0, 0);

    // Segmentation
    std::vector<std::vector<Point>> contours1;
    vector<Vec4i> hierachy1;
    findContours(dilated_cpy, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
    Mat dst = Mat::zeros(imgDilated.size(), CV_8UC3);

    if (!contours1.empty())
    {
        for (int i = 0; i < contours1.size(); i++)
        {
            Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
            drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
        }
    }

    // Bounding box detection
    for (int i = 0; i < contours1.size(); i++)
    {
        rect = boundingRect(contours1[i]);

        float ratio = ((float)rect.width / (float)rect.height);

        bool too_small = rect.width < 40 || rect.height < 20;
        bool too_big = rect.width > 300 || rect.height > 150;
        bool ratio_big = ratio > 20;
        bool ratio_small = ratio < 2;
        bool outsideROI = rect.x < 0.1 * imgGrey.cols || rect.x > 0.9 * imgGrey.cols || rect.y < 0.1 * imgGrey.rows || rect.y > 0.9 * imgGrey.rows;

        if (too_small || too_big || outsideROI || ratio_big || ratio_small)
        {
            drawContours(dilated_cpy, contours1, i, black, -1, 8, hierachy1);
        }
        else
        {
            plate = imgGrey(rect);

            //Mat addImg = imgDilated(rect);
            //Mat addImgGrey = imgGrey(rect);

            //float focusAmount = 0.3;
            //Point2i p1 = Point2i(rect.x + rect.width * focusAmount, rect.y + rect.height * focusAmount);
            //Point2i p2 = Point2i(rect.x + rect.width - (rect.width * focusAmount), rect.y + rect.height - (rect.height * focusAmount));
            //Rect focusRect(p1, p2);

            //float edgeFill = FillRatio(imgEdge(focusRect));
            //float fill = FillRatio(addImg);
            //if (fill > 0.7 && edgeFill > 0.15)
            //{
            //    list.push_back(ImgInfo(addImg, addImgGrey, ratio, fill));
            //    imshow("ye " + i, imgEdge(focusRect));
            //    std::cout << "Edge " << i << ": " << edgeFill << std::endl;
            //}
        }
    }

    imshow("Filtered image", dilated_cpy);

    if ((plate.rows != 0 && plate.cols != 0))
    {
        imshow("Plate", plate);
    }
    else
    {
        std::cout << "No valid plate found" << std::endl;
    }

    int otsuTh = OTSU(plate);
    Mat binPlate = GreyToBinary(plate, otsuTh);

    imshow("Binary plate", binPlate);

    Mat plateCpy;
    plateCpy = binPlate.clone();
    vector<vector<Point>> contours2;
    vector<Vec4i> hierarchy2;
    findContours(plateCpy, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
    Mat dst2 = Mat::zeros(binPlate.size(), CV_8UC3);

    if (!contours2.empty())
    {
        for (int i = 0; i < contours2.size(); i++)
        {
            Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
            drawContours(dst2, contours2, i, colour, -1, 8, hierarchy2);
        }
    }

    imshow("Segmented plate", dst2);

    vector<CharBox> charList;
    char* outText;
    for (int i = 0; i < contours2.size(); i++)
    {
        rect = boundingRect(contours2[i]);
        float ratio = ((float)rect.width / (float)rect.height);
        bool min_ratio = ratio < 0.5;
        bool max_ratio = ratio > 2;

        if (rect.height < 5 || min_ratio || max_ratio)
        {
            drawContours(plateCpy, contours2, i, black, -1, 8, hierarchy2);
        }
        else
        {
            Mat plateChar;
            plateChar = binPlate(rect);

            Mat charBordered = CopyWithBorder(plateChar, 10);

            imshow("char" + to_string(i), charBordered);

            api->SetImage(charBordered.data, charBordered.cols, charBordered.rows, charBordered.channels(), charBordered.step1());
            outText = api->GetUTF8Text();

            std::cout << "-------------" << std::endl;
            std::cout << "Char " << to_string(i) << ": " << outText << std::endl;
            std::cout << "Pos " << to_string(i) << ": " << rect.x << std::endl;
            std::cout << "-------------" << std::endl;

            CharBox currChar(outText, rect.x, rect.y);
            charList.push_back(currChar);
        }
    }

    char* finalCharPlate = SortedCharBox(charList);
    std::cout << finalCharPlate << std::endl;

    waitKey();

    //api->End();
    //delete api;
    //delete[] outText;
}