#include <iostream>
#include "core.h"

using namespace cv;
using namespace cvext;
using namespace std;

#define NULL_ID -1
struct IDGrid
{
    char value;
    int id;

    IDGrid()
    {
        value = '0';
        id = NULL_ID;
    }

    IDGrid(char value, int id)
    {
        this->value = value;
        this->id = id;
    }
};

vector<vector<IDGrid>> convertToID(const vector<vector<char>>& grid)
{
    vector<vector<IDGrid>> output;

    int rows = grid.size();
    int cols = grid[0].size();
    for (int i = 0; i < rows; i++)
    {
        vector<IDGrid> col;
        for (int j = 0; j < cols; j++)
        {
            col.push_back(IDGrid(grid[i][j], NULL_ID));
        }
        output.push_back(col);
    }

    return output;
}

vector<vector<IDGrid>> makeIdGrid(vector<vector<char>> grid)
{
    vector<vector<IDGrid>> idgrid;

    int rows = grid.size();
    int cols = grid[0].size();

    for (int i = 0; i < rows; i++)
    {
        vector<IDGrid> col;
        for (int j = 0; j < cols; j++)
        {
            col.push_back(IDGrid(grid[i][j], NULL_ID));
        }
        idgrid.push_back(col);
    }

    int id = -1;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j <cols; j++)
        {
            IDGrid curr = idgrid[i][j];
            if (curr.value != '1')
            {
                continue;
            }

            int l_index = j - 1;
            int t_index = i - 1;

            bool l_valid = l_index >= 0 && l_index < rows;
            bool t_valid = t_index >= 0 && t_index < cols;

            IDGrid l_neighbour;
            IDGrid t_neighbour;

            // has left neighbour and no top neighbour
            if (l_valid && !t_valid)
            {
                l_neighbour = idgrid[i][l_index];
                if (l_neighbour.id == NULL_ID)
                {
                    id++;
                    curr.id = id;
                }

                if (l_neighbour.id != NULL_ID)
                {
                    curr.id = l_neighbour.id;
                }
            }

            // has top neighbour and no left neighbour
            if (t_valid && !l_valid)
            {
                t_neighbour = idgrid[t_index][j];
                if (t_neighbour.id == NULL_ID)
                {
                    id++;
                    curr.id = id;
                }

                if (t_neighbour.id != NULL_ID)
                {
                    curr.id = t_neighbour.id;
                }
            }

            // has no neighbours
            if (!t_valid && !l_valid)
            {
                id++;
                curr.id = id;
            }

            // has both neighbours
            if (l_valid && t_valid)
            {
                l_neighbour = idgrid[i][l_index];
                t_neighbour = idgrid[t_index][j];

                if (l_neighbour.id != NULL_ID && t_neighbour.id == NULL_ID)
                {
                    curr.id = l_neighbour.id;
                }
                if (l_neighbour.id == NULL_ID && t_neighbour.id != NULL_ID)
                {
                    curr.id = t_neighbour.id;
                }
                if (l_neighbour.id == NULL_ID && t_neighbour.id == NULL_ID)
                {
                    id++;
                    curr.id = id;
                }
                if (l_neighbour.id != NULL_ID && t_neighbour.id != NULL_ID)
                {
                    if (l_neighbour.id <= t_neighbour.id)
                    {
                        curr.id = l_neighbour.id;
                    }
                    else
                    {
                        curr.id = t_neighbour.id;
                    }
                }
            }
        }
    }
    return idgrid;
}

void printIDGrid(vector<vector<IDGrid>> grid)
{
    int rows = grid.size();
    int cols = grid[0].size();

    for (size_t i = 0; i < rows; i++)
    {
        if (i != 0)
        {
            std::cout << std::endl;
        }
        for (size_t j = 0; j < cols; j++)
        {
            std::cout << "[" << grid[i][j].id << "]";
        }
    }
}

struct ImgInfo
{
    Mat imgBin;
    Mat imgGrey;
    float ratio;
    float fill;

    int rows;
    int cols;

    ImgInfo(Mat imgBin, Mat imgGrey, float ratio, float fill)
    {
        this->imgBin = imgBin;
        this->imgGrey = imgGrey;
        this->ratio = ratio;
        this->fill = fill;

        rows = imgBin.rows;
        cols = imgBin.cols;
    };
};

#define NUM_IMAGES 1

int main()
{
    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
    if (api->Init("tesseract-ocr\\tessdata", "eng"))
    {
        std::cout << "Could not initialize Tesseract!" << std::endl;
        exit(1);
    }

    //for (int i = 1; i <= NUM_IMAGES; i++)
    //{
        //String path = "content\\" + to_string(i) + ".jpg";
        String path = "content\\10.jpg";
        Mat imgBin = imread(path);
        Mat imgGrey = RgbToGrey(imgBin);
        Mat imgAvg = Average(imgGrey, 1);
        Mat imgEdge = Edge(imgAvg, 50);
        Mat imgErosion = Erosion(imgEdge, 1);
        Mat imgDilated1 = Dilation(imgErosion, 7);
        Mat imgDilated = HorizontalDilation(imgDilated1, 12);

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

        vector<ImgInfo> list;

        // Bounding box detection
        for (int i = 0; i < contours1.size(); i++)
        {
            rect = boundingRect(contours1[i]);

            float ratio = ((float)rect.width / (float)rect.height);

            bool too_small = rect.width < 40 || rect.height < 20;
            bool too_big = rect.width > 400 || rect.height > 200;
            bool ratio_big = ratio > 20;
            bool ratio_small = ratio < 2;
            bool outsideROI = rect.x < 0.1 * imgGrey.cols || rect.x > 0.9 * imgGrey.cols || rect.y < 0.1 * imgGrey.rows || rect.y > 0.9 * imgGrey.rows;

            if (too_small || too_big || outsideROI || ratio_big || ratio_small)
            {
                drawContours(dilated_cpy, contours1, i, black, -1, 8, hierachy1);
            }
            else
            {
                Mat addImg = imgDilated(rect);
                Mat addImgGrey = imgGrey(rect);

                float focusAmount = 0.3;
                Point2i p1 = Point2i(rect.x + rect.width * focusAmount, rect.y + rect.height * focusAmount);
                Point2i p2 = Point2i(rect.x + rect.width - (rect.width * focusAmount), rect.y + rect.height - (rect.height * focusAmount));
                Rect focusRect(p1, p2);

                float edgeFill = FillRatio(imgEdge(focusRect));
                float fill = FillRatio(addImg);
                if (fill > 0.7 && edgeFill > 0.15)
                {
                    list.push_back(ImgInfo(addImg, addImgGrey, ratio, fill));
                    imshow("ye " + i, imgEdge(focusRect));
                    std::cout << "Edge " << i << ": " << edgeFill << std::endl;
                }
            }
        }

        if (list.size() > 0)
        {
            for (size_t i = 0; i < list.size(); i++)
            {
                if (list[i].rows != 0 && list[i].cols != 0)
                {
                    std::cout << "Fill " << i << ": " << list[i].fill << std::endl;
                    std::cout << "Ratio " << i << ": " << list[i].ratio << std::endl;
                    std::cout << "Cols " << i << ": " << list[i].cols << std::endl;
                    std::cout << "Rows " << i << ": " << list[i].rows << std::endl;
                    imshow("grey " + i, list[i].imgGrey);
                    imshow("bin " + i, list[i].imgBin);
                }
            }

            std::cout << "Possible count: " << list.size() << std::endl;
        }
        else
        {
            std::cout << "Didnt find possible plates" << std::endl;
        }
    //}
    //api->SetImage(plate.data, plate.cols, plate.rows, plate.channels(), plate.step1());

    //char* outText;
    //outText = api->GetUTF8Text();

    //std::cout << "-------------" << std::endl;
    //std::cout << outText << std::endl;
    //std::cout << "-------------" << std::endl;

    waitKey();

    //api->End();
    //delete api;
    //delete[] outText;
}