#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	imshow("src", src);
	
	Mat dst;
	for (int ksize = 3; ksize <= 7; ksize += 2) { // 3부터 7까지 2씩 증가하므로 3, 5, 7
		blur(src, dst, Size(ksize, ksize)); // 각각의 for문에 대하여 ksize x ksize 만큼의 블러링 수행

		String desc = format("Mean: %dx%d", ksize, ksize); // 어떤 크기의 필터인지 영상 위에 문자열로 나타냄
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, 
			Scalar(255), 1, LINE_AA);

		imshow("dst", dst);
		waitKey(); // 키보드 키를 기다림
	}
}