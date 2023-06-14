#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("milkdrop.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat src_bin;
	threshold(src, src_bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	vector<vector<Point>> contours;
	findContours(src_bin, contours, RETR_LIST, CHAIN_APPROX_NONE); // 모든 외곽선을 다 검출

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	for (unsigned i = 0; i < contours.size(); i++) { 
	// RETR_LIST를 사용했으므로 검출된 외곽선을 0번 부터 contours.size 까지 차례대로 그리면 모든 외곽선을 그릴 수 있음
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, i, color, 1, LINE_8);
	}

	imshow("src", src);
	imshow("src_bin", src_bin);
	imshow("dst", dst);
	waitKey();
}