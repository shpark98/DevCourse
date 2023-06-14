#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc < 2) {
		cerr << "Usage: adjmean.exe <filename>" << endl;
		return -1;
	}

	Mat src = imread(argv[1], IMREAD_GRAYSCALE); // argv[1] 영상을 그레이스케일 형태로 불러와 src에 저장

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	int s = 0;
	
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			s += src.at<uchar>(j, i); 	// 입력 영상인 src의 평균 밝기 구하기
		}
	}

	int m = s / (src.rows * src.cols);
	//int m = sum(src)[0] / src.total();
	//int m = mean(src)[0];

	cout << "Mean value: " << m << endl;

	Mat dst = src + (128 - m); 	// 평균 밝기가 128이 되도록 밝기 보정하기
	
	imshow("src", src);
	imshow("dst", dst);

	waitKey();
}