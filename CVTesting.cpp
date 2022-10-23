#include <iostream>
#include "core.h"

using namespace cv;
using namespace cvext;
using namespace std;

#define NUM_IMAGES 19

int main()
{
    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
    if (api->Init("tesseract-ocr\\tessdata", "eng", tesseract::OEM_DEFAULT))
    {
        std::cout << "Could not initialize Tesseract!" << std::endl;
        exit(1);
    }

    api->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    api->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK);

    //String path = "content\\"+to_string(i)+".jpg";
    String path = "content\\19.jpg";
    Mat imgBin = imread(path);

    /*
    * FINDING LICENSE PLATE IN IMAGE
    */
    Mat imgGrey = RgbToGrey(imgBin);
    Mat imgAvg = Average(imgGrey, 1);
    Mat imgEdge = Edge(imgAvg, 60);
    Mat imgErosion = Erosion(imgEdge, 1);
    Mat imgDilated = Dilation(imgErosion, 15);
    imgDilated = HorizontalDilation(imgDilated, 5);

    imshow("Grey", imgGrey);
    imshow("Final", imgDilated);

    Mat dilated_cpy = imgDilated.clone();
    Mat plate;
    Rect rect;
    Scalar black = CV_RGB(0, 0, 0);

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

    for (int i = 0; i < contours1.size(); i++)
    {
        rect = boundingRect(contours1[i]);

        float ratio = ((float)rect.width / (float)rect.height);

        bool too_small = rect.width < 50 || rect.height < 20;
        bool too_big = rect.width > 350 || rect.height > 80;
        bool ratio_small = ratio < 1.5;
        bool outsideROI = rect.x < 0.1 * imgGrey.cols || rect.x > 0.9 * imgGrey.cols || rect.y < 0.1 * imgGrey.rows || rect.y > 0.9 * imgGrey.rows;

        if (too_small || too_big || outsideROI || ratio_small)
        {
            drawContours(dilated_cpy, contours1, i, black, -1, 8, hierachy1);
        }
        else
        {
            plate = imgGrey(rect);
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
        return 0;
    }

    /*
    * FINDING CHARACTERS IN LICENSE PLATE
    */
    // Plate grey processing
    resize(plate, plate, Size(plate.size().width * 4, plate.size().height * 4), 0, 0, cv::INTER_LINEAR);
    int otsuTh = OTSU(plate);
    imshow("Grey plate", plate);

    int finalOtsu = otsuTh + 60;
    if (finalOtsu > 220)
    {
        finalOtsu = 220;
    }
    Mat binPlate = Step(plate, finalOtsu, 255);
    imshow("Processed plate", binPlate);

    // Segmenting characters in plate
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

    for (int i = 0; i < contours2.size(); i++)
    {
        rect = boundingRect(contours2[i]);
        float ratio = ((float)rect.width / (float)rect.height);
        bool min_ratio = ratio < 0.5;
        bool max_ratio = ratio > 2;

        if (rect.height < 20 || rect.height > 100 || min_ratio || max_ratio)
        {
            drawContours(plateCpy, contours2, i, black, -1, 8, hierarchy2);
        }
        else
        {
            Mat plateChar;

            // Char grey
            plateChar = plate(rect);
            resize(plateChar, plateChar, Size(plateChar.size().width * 4, plateChar.size().height * 4), 0, 0, cv::INTER_LINEAR);
            plateChar = Average(plateChar, 3);
            plateChar = Step(plateChar, 200, 255);
            plateChar = CopyWithBorder(plateChar, 15);

            api->SetImage(plateChar.data, plateChar.cols, plateChar.rows, 1, plateChar.step);
            char* outText;
            outText = api->GetUTF8Text();

            CharBox currChar(*outText, rect.x, rect.y);

            charList.push_back(currChar);
            imshow("Char" + to_string(i), plateChar);
        }
    }

    /*
    * SORTING CHARACTERS AND PRINTING LICENSE PLATE
    */
    char* output = SortedCharBox(charList); /*= SortedCharBox(charList);*/

    std::cout << "Plate: ";
    for (size_t i = 0; i < charList.size(); i++)
    {
        std::cout << output[i];
    }
    std::cout << std::endl;
    waitKey();

    waitKey();
    api->End();
    delete api;
    delete[] output;
}