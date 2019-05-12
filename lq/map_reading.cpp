#pragma once
#include "map_reading.h"

//These three vars are used by onMouse and getPosByMaus.
int count_maus_clicks = 0;
std::string winName;
cv::Point2f result_of_onMouse[4];

static void onMouse(int event, int x, int y, int, void*) {
	if (event == cv::EVENT_LBUTTONDOWN) {
		if (count_maus_clicks <= 4) {
			result_of_onMouse[count_maus_clicks].x = (float)x;
			result_of_onMouse[count_maus_clicks].y = (float)y;
			count_maus_clicks++;
			if (count_maus_clicks < 4)
				std::cout << "\nPlease click point" << count_maus_clicks + 1 << " in window \"" << winName << "\".";
			else if (count_maus_clicks == 4)
				std::cout << "\nInput ends.Press any button to continue." << std::endl;
		}
	}
}

cv::Point2f* getPosByMaus(const std::string windowName) {
	winName = windowName;
	std::cout << "\nPlease click point1 in window \"" << winName << "\".";
	cv::setMouseCallback(windowName, onMouse, 0);
	cv::waitKey(0);
	return result_of_onMouse;
}

//The two vars below is used by rescaleWarp to determine if 
//it's the first time a pic needs to warp, with the same warp_mat.
bool first_time_warp = true;
cv::Mat warp_mat;

void firstRescaleWarp(cv::Point2f* startPoints, cv::Mat picture) {
	//define the warp destination points as described in the .h file
	cv::Point2f destPoints[4];
	destPoints[0] = cv::Point2f(0.f, 0.f);
	destPoints[1] = cv::Point2f(picture.cols - 1.f, 0.f);
	destPoints[2] = cv::Point2f(0.f, picture.rows - 1.f);
	destPoints[3] = cv::Point2f(picture.cols - 1.f, picture.rows - 1.f);

	//get the transform mat
	warp_mat = getPerspectiveTransform(startPoints, destPoints);
}

cv::Mat rescaleWarp(cv::Point2f* startPoints, cv::Mat picture)
{
	if (first_time_warp) {
		firstRescaleWarp(startPoints, picture);
		first_time_warp = false;
	}

	//create a blank result_picture
	cv::Mat result_picture = cv::Mat::zeros(picture.rows, picture.cols, picture.type());
	
	//do the warp
	cv::warpPerspective(picture, result_picture, warp_mat, result_picture.size());
	return result_picture;
}

cv::Mat rescaleWarp(cv::Mat picture) {
	if (first_time_warp) {
		std::cerr << "No warp rule made yet!";
		return cv::Mat::zeros(picture.rows, picture.cols, picture.type());
	}

	//create a blank result_picture
	cv::Mat result_picture = cv::Mat::zeros(picture.rows, picture.cols, picture.type());

	//do the warp
	cv::warpPerspective(picture, result_picture, warp_mat, result_picture.size());

	return result_picture;
}


//Initialization for the video window
void videoOnCreate(const std::string videoWinName, int* thresholdPtr) {
	cv::createTrackbar("Threahold", videoWinName, thresholdPtr, 255, 0, 0);
}


//Map border auto detection, color is specified by two scalars.
//Returns the four vertices of the map.
cv::Point2f* getPosByColoredShapes(cv::Mat rgbPic, cv::Scalar lowerS, cv::Scalar upperS) {

	cv::Mat hsvPic;
	cvtColor(rgbPic, hsvPic, cv::COLOR_BGR2HSV);

	cv::Mat mask;
	cv::Mat result = cv::Mat::zeros(rgbPic.size(), CV_8UC3);

	//Two ways of detection: auto, and manual(trackBars).
	//The latter is for parameter adjustments.
	//	//#1
	//cv::inRange(hsvPic, lowerS, upperS, mask);

	////apply the mask to the rgbPic to take what we need
	//for (int r = 0; r < rgbPic.rows; r++)
	//{
	//	for (int c = 0; c < rgbPic.cols; c++)
	//	{
	//		if (mask.at<uchar>(r, c) == 255)
	//		{
	//			result.at<cv::Vec3b>(r, c) = rgbPic.at<cv::Vec3b>(r, c);
	//		}
	//	}
	//}//#1 ends

		//#2
	cv::namedWindow("Manual param selection");
	int L = 0, U = 255, *lH = &L, *uH = &U;
	while (1) {
		cv::createTrackbar("lowerH", "Manual param selection", lH, 255, 0, 0);
		cv::createTrackbar("upperH", "Manual param selection", uH, 255, 0, 0);
		cv::inRange(hsvPic, cv::Scalar(L, lowerS[1], lowerS[2]), cv::Scalar(U, upperS[1], upperS[2]), mask);
		
		result = cv::Mat::zeros(rgbPic.size(), CV_8UC3);

		for (int r = 0; r < rgbPic.rows; r++){
			for (int c = 0; c < rgbPic.cols; c++){
				if (mask.at<uchar>(r, c) == 255){
					result.at<cv::Vec3b>(r, c) = rgbPic.at<cv::Vec3b>(r, c);
				}
			}
		}

		cv::imshow("Manual param selection", result);
		if (cv::waitKey(20) >= 5)
			break;
	}//#2 ends

	//Erode and then dilate, erases noise
	cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
	cv::erode(result, result, erodeElement);
	cv::dilate(result, result, dilateElement);

	//er zhi hua
	threshold(result, result, 0, 255, cv::THRESH_BINARY);

	//detect contours
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat grayResult;
	cv::cvtColor(result, grayResult, cv::COLOR_RGB2GRAY);
	cv::findContours(grayResult, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::cout << contours.size();
	cv::Point2f *points = new cv::Point2f[contours.size()];
	cv::Rect rect;
	cv::Mat display = rgbPic;
	for (int nthC = 0; nthC < contours.size(); nthC++) {
		rect = cv::boundingRect(contours[nthC]);
		cv::rectangle(display, rect, cv::Scalar(0, 255, 0), 2);
		points[nthC] = cv::Point2f(rect.x + 0.5 * rect.width, rect.y + 0.5 * rect.height);
	}

	cv::namedWindow("rectShow");
	cv::imshow("rectShow", display);

	cv::Point2f pointTmp = points[3]; points[3] = points[0]; points[0] = pointTmp;
	pointTmp = points[2]; points[2] = points[1]; points[1] = pointTmp;


	return points;
}


cv::Mat thin(cv::Mat src,int iterations) {
	cv::Mat dst, dstdat;
	src.copyTo(dst);
	src.copyTo(dstdat);
	for (int n = 0; n < iterations; n++)
		for (int s = 0; s <= 1; s++) {
			dst.copyTo(dstdat);
			for (int i = 0; i < src.rows; i++)
				for (int j = 0; j < src.cols; j++)
					if (dstdat.at<uchar>(i, j)) {
						int a = 0, b = 0;
						int d[8][2] = { {-1, 0}, {-1, 1}, {0, 1}, {1, 1},
						 {1, 0}, {1, -1}, {0, -1}, {-1, -1} };
						int p[8];
						p[0] = (i == 0) ? 0 : dstdat.at<uchar>(i - 1, j);
						for (int k = 1; k <= 8; k++) {
							if (i + d[k % 8][0] < 0 || i + d[k % 8][0] >= src.rows ||
								j + d[k % 8][1] < 0 || j + d[k % 8][1] >= src.cols)
								p[k % 8] = 0;
							else p[k % 8] = dstdat.at<uchar>(i + d[k % 8][0], j + d[k % 8][1]);
							if (p[k % 8]) {
								b++;
								if (!p[k - 1]) a++;
							}
						}
						if (b >= 2 && b <= 6 && a == 1)
							if (!s && !(p[2] && p[4] && (p[0] || p[6])))
								dst.at<uchar>(i, j) = 0;
							else if (s && !(p[0] && p[6] && (p[2] || p[4])))
								dst.at<uchar>(i, j) = 0;
					}
		}
	return dst;
}


int new_count_maus_clicks = 0;
std::string new_winName;
std::vector<cv::Point2f> strtEndResult;
//new OnMouse for getStrtEnd()
static void onMouse2(int event, int x, int y, int, void*) {
	if (event == cv::EVENT_LBUTTONDOWN) {
		if (new_count_maus_clicks < 2) {
			cv::Point2f pointTmp((float)x, (float)y);
			strtEndResult.push_back(pointTmp);
			if (new_count_maus_clicks == 0)
				std::cout << "Please input end point;\n";
			else
				std::cout << "Input ends.Press any button to continue.\n";
			new_count_maus_clicks++;
		}
	}
}
//function for getting the starting points and ending points by mause clicks
std::vector<cv::Point2f> getStrtEnd(const std::string windowName) {
	new_winName = windowName;
	std::cout << "\nWorking on " << new_winName << ".\nPlease input start point; \n";
	cv::setMouseCallback(windowName, onMouse2, 0);
	cv::waitKey(0);
	return strtEndResult;
}