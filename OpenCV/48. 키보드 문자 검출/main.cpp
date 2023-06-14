#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("keyboard.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat src_bin;
	threshold(src, src_bin, 0, 255, THRESH_BINARY | THRESH_OTSU); 

	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(src_bin, labels, stats, centroids);
	cout << cnt << endl;

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (int i = 1; i < cnt; i++) {
		int* p = stats.ptr<int>(i); 
		// stats 행렬에서 i번째 행의 정보 5개 (x, y, width, height, label 면적)를 p로 받음 

		if (p[4] < 20) continue; // 레이블의 픽셀 개수가 너무 작으면 그림을 그리지 않게함

		rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255)); 
		// 충분히 큰 객체에 대해서만 사각형을 그림
		
		// Mat crop = src(Rect(p[0], p[1], p[2], p[3])); // 표시한 네모를 crop 해서 crop 객체에 저장
		// imshow("crop", crop);
		// waitKey();
	}

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
