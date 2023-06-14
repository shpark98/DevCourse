#include <iostream>
#include <algorithm>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int pos_hue1 = 50, pos_hue2 = 80, pos_sat1 = 150, pos_sat2 = 255;
Mat src, src_hsv, dst, mask;

void on_hsv_changed(int, void*) // 트랙바를 움직일때 자동으로 호출됨
{
	Scalar lowerb(pos_hue1, pos_sat1, 0); // 하한값 value의 상한값은 0으로 지정
	Scalar upperb(pos_hue2, pos_sat2, 255); // 상한값 value의 상한값은 255로 지정 value는 어떤값이든 상관없이 H와 S의 값만 만족하면 찾겠다 라는 의미
	inRange(src_hsv, lowerb, upperb, mask); // 위에서 지정한 범위에 만족하는 색상은 흰색255 로 설정되고 그렇지 않은 부분은 검정색 0 으로 설정한 것을 mask에 저장

	cvtColor(src, dst, COLOR_BGR2GRAY); // 컬러영상 src 영상을 그레이영상 dst로 만듬
	cvtColor(dst, dst, COLOR_GRAY2BGR); 
	// 그레이영상 dst를 다시 BGR로 바꾼다해서 이미 손상된 컬러 정보가 회복되진 않음.
	// 따라서 그레이로 보이는 것은 똑같은데 dst가 가진 채널 개수가 1개에서 3개로 늘어남(CV_8UC3 타입으로 변환)
	src.copyTo(dst, mask); // mask 영상에서 검정색이 아닌 부분(픽셀값이 0이 아닌부분)에 대해서 src영상을 dst 영상으로 Copy

	imshow("mask", mask);
	imshow("dst", dst);
}

int main()
{
	src = imread("candies.png", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	cvtColor(src, src_hsv, COLOR_BGR2HSV);

	namedWindow("src");
	namedWindow("mask");
	namedWindow("dst");

	imshow("src", src);

	createTrackbar("Lower Hue", "dst", &pos_hue1, 179, on_hsv_changed);
	createTrackbar("Upper Hue", "dst", &pos_hue2, 179, on_hsv_changed);
	createTrackbar("Lower Sat", "dst", &pos_sat1, 255, on_hsv_changed);
	createTrackbar("Upper Sat", "dst", &pos_sat2, 255, on_hsv_changed); 
	// Trackbar 4개를 붙임
	
	on_hsv_changed(0, 0);
	// 프로그램이 실행되자마자 on_hsv_changed가 실행될 수 있게 강제로 작성함

	waitKey();
}
