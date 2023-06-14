#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src1 = imread("lenna256.bmp", IMREAD_GRAYSCALE);
	Mat src2 = imread("square.bmp", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	if (src1.size() != src2.size() || src1.type() != src2.type()) {
		cerr << "The images are different in size or type!" << endl;
		return -1;
	}

	imshow("src1", src1);
	imshow("src2", src2);

	Mat dst1, dst2, dst3, dst4;

	add(src1, src2, dst1); // 덧셈 연산
	addWeighted(src1, 0.5, src2, 0.5, 0, dst2); // 알파와 베타값이 0.5이므로 평균
	subtract(src1, src2, dst3); // 뺄셈 연산
	absdiff(src1, src2, dst4); // 차이 연산

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);
	imshow("dst4", dst4); 
	waitKey();
}