#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("contours.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE); // 외곽선 검출을 수행
	// 입력 영상 src가 이진화 형태로 되어있어서 바로 입력 영상으로 줬음
	// contours 는 vector<vector<Point>> 타입의 형태를 지정
	// hierarchy 는 	vector<Vec4i> 로 지정
	// RETR_CCOMP 를 입력하여 부모 자식 관계를 2단계로만..
	// CHAIN_APPROX_NONE 근사화 작업은 따로 하지 않음
	

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) { // 외곽선을 실제로 그려서 화면에 보여주는 코드
	// 0번 외곽선부터 시작해서 외곽선이 0번보다 클 동안 for 문이 계속 돌음
	// 현재 인덱스에 해당하는 외곽선의 [0]인 next의 정보로 idx 업데이트

		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, idx, color, 2, LINE_8, hierarchy);
		// 그리고 특정 색으로 drawContours 실행
	}

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}