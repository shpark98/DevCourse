#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void on_level_change(int pos, void* userdata);

int main(void)
{
	Mat img = Mat::zeros(400, 400, CV_8UC1);
	// main 함수에 있는 어떤 영상 데이터를 mouse 콜백 함수나 trackbar 콜백 함수에 전달하려면 img를 전역 변수형식으로 선언하거나 userdata 형태 타입인 void 포인터 형식으로 주소값을 전달해야 함

	namedWindow("image");
	createTrackbar("level", "image", 0, 16, on_level_change, (void*)&img);

	imshow("image", img);
	waitKey();
}

void on_level_change(int pos, void* userdata)
{
	Mat img = *(Mat*)userdata;

	img.setTo(pos * 16);
	imshow("image", img);
}