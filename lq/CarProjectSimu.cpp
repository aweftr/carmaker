// CarProjectSimu.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include "map_reading.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main()
{
	Mat first_pic, frame;

	//Open camera
	VideoCapture cam;
	int deviceID = 0;
	int apiID = CAP_ANY;
	cam.open(deviceID, apiID);

	if (!cam.isOpened()) {
		std::cerr << "ERROR! Unable to open camera\n";
		return -1;
	}

	//Open window showing the first picture
	cam.read(first_pic);
	namedWindow("Vanilla perspective");
	imshow("Vanilla perspective", first_pic);
	waitKey(1000);


	//Show realtime video with the ordered perspective change.

	//Point2f *startp = getPosByMaus("Point selection");
	Point2f *startp = getPosByColoredShapes(first_pic, Scalar(0, 43, 46), Scalar(20, 255, 255));

	namedWindow("Changed perspective");
	int* thresPtr = new int(0); 
	videoOnCreate("Changed perspective", thresPtr);

	Mat tmp, thresTmp;
	while (1) {
		cam.read(frame);

		if (frame.empty()) {
			std::cerr << "Empty frame captured!";
			break;
		}

		tmp = rescaleWarp(startp, frame);

		//cvtColor(tmp, thresTmp, COLOR_RGB2GRAY);
		//threshold(thresTmp, thresTmp, *thresPtr, 255, THRESH_BINARY);
		//adaptiveThreshold(thresTmp, thresTmp, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, *thresPtr);

		imshow("Changed perspective", tmp);
		if (waitKey(20) >= 5)
			break;
	}

	cam.release();
	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
