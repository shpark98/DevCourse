#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("coins.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat dst1, dst2;
	cvtColor(src, dst1, COLOR_GRAY2BGR);
	cvtColor(src, dst2, COLOR_GRAY2BGR);

	TickMeter tm;

	// HOUGH_GRADIENT

	Mat blr;
	GaussianBlur(src, blr, Size(), 1.0); // Gaussian Blur 한 결과 blr을 만듬

	tm.start();

	vector<Vec3f> circles1;
	HoughCircles(blr, circles1, HOUGH_GRADIENT, 1, 10, 150, 30, 10, 50); // HOUGH_GRADIENT 방법 사용
	// 가우시안 블러작업을 한 blr 영상을 넣어줌, HoughCircles 함수가 입력영상에서 각각의 엣지 픽셀 위치를 상당히 민감하게 반응하는 부분이 있어 오리지널 영상을 그대로 사용할 경우 아주 미세한 차이로 원을 검출하지 못할 수 있음 -> 블러를 통해 영상을 부드럽게 만듬
	// 10 : 중심점의 최소 거리
	

	tm.stop();
	cout << "HOUGH_GRADIENT:     " << tm.getTimeMilli() << "ms." << endl;

	for (size_t i = 0; i < circles1.size(); i++) {
		Vec3i c = circles1[i];
		circle(dst1, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 2, LINE_AA);
	}

	// HOUGH_GRADIENT_ALT

	tm.reset();
	tm.start();

	vector<Vec3f> circles2;
	HoughCircles(src, circles2, HOUGH_GRADIENT_ALT, 1.5, 10, 300, 0.9, 10, 50); // HOUGH_GRADIENT_ALT 방법 사용  
	// 축적배율의 크기를 1.5배 작게.. 위 HOUGH_GRADIENT 방법에서 블러링된 영상을 입력으로 준 것과 비슷한 느낌
	// 너무 민감하게 검출하지 않고 약간의 잡음이 포함되어있거나 원이 살짝 튀어도 원을 잘 검출할 수 있게 해줌 (적정: 1.5~2.0)
	// 10은 중심점의 최소 거리
	// 1은 완벽한 원을 검출, 0.9는 살짝 찌그러진 것도 검출, 0.8은 더 찌그러진 것도 검출

	tm.stop();
	cout << "HOUGH_GRADIENT_ALT: " << tm.getTimeMilli() << "ms." << endl;

	for (size_t i = 0; i < circles2.size(); i++) { // 결과 표시
		Vec3i c = circles2[i];
		circle(dst2, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	waitKey();
}