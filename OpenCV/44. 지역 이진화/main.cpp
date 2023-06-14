#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("rice.png", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat dst1;
	threshold(src, dst1, 0, 255, THRESH_BINARY | THRESH_OTSU); // 자동으로 threshold를 설정하여 전역 이진화

	int bw = src.cols / 4; 
	int bh = src.rows / 4; // 128 * 128 짜리 16 구역으로 나뉘어서 threshold 함수를 실행

	Mat dst2 = Mat::zeros(src.rows, src.cols, CV_8UC1);

	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			Mat src_ = src(Rect(x*bw, y*bh, bw, bh)); // 원본영상의 부분영상을 추출
			Mat dst_ = dst2(Rect(x*bw, y*bh, bw, bh)); // 참조로 결과영상을 받아옴
			threshold(src_, dst_, 0, 255, THRESH_BINARY | THRESH_OTSU); 
			// 원본영상의 부분영상 src_의 이진화 결과를 dst_에 저장 
			// dst_를 dst2 참조로 결과영상을 받아와서 threshold 함수를 dst_에 실행해도 dst2에 갱신됨 
		}
	}

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	waitKey();
}