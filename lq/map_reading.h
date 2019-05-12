#pragma once
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>

//---------------------------------------------------------------------
//Get 4 points specified by left mouse clicks on the window 'windowName'.
//Let's suppose p1 be the top-left corner, p2 be the top-right corner
//p3 be the bottom-left corner and p4 be the bottom-right corner.
cv::Point2f* getPosByMaus(std::string windowName);
//---------------------------------------------------------------------


//---------------------------------------------------------------------
//Gets 4 points specified by centers of colored shapes whose colors lie
//in the 3-dimentional HSV color space specified by the two vertices, 
//namely scalars. RgbPic must be in rgb form. No default detection color.
cv::Point2f* getPosByColoredShapes(cv::Mat rgbPic, cv::Scalar lowerS, cv::Scalar upperS);
//---------------------------------------------------------------------


//---------------------------------------------------------------------
//Reads in 4 points(in param 'startPoints'), and returns the transformed
//picture of that corresponding new perspective.
cv::Mat rescaleWarp(cv::Point2f* startPoints, cv::Mat picture);
//---------------------------------------------------------------------


//---------------------------------------------------------------------
//Does the initializing such as trackBars on the creation of the video page.
void videoOnCreate(const std::string videoWinName, int* thresholdPtr);
//---------------------------------------------------------------------


//---------------------------------------------------------------------
//Thinning function from the instruction book, tailored to this version
//of openCV. Tailored to current version now, and is fully functioning.
cv::Mat thin(cv::Mat src, int iterations);
//---------------------------------------------------------------------


//---------------------------------------------------------------------
//Gets start point and end point by mouse. Simillar to getPosByMaus().
std::vector<cv::Point2f> getStrtEnd(const std::string windowName);
//---------------------------------------------------------------------