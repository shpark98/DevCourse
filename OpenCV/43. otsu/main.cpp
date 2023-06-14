#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("rice.png", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat dst;
	double th = threshold(src, dst, 0, 255, THRESH_BINARY | THRESH_OTSU); // THRESH_OTSU를 사용
	// 반환하는 이진화 임계값을 th로 받음
	// THRESH_BINARY | THRESH_OTSU 를 THRESH_OTSU으로 대체해도 됨
	
	cout << "Otsu threshold value is " << th << "." << endl;

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}