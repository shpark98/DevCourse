#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	TickMeter tm;
	tm.start();

	Mat dst1;
	GaussianBlur(src, dst1, Size(), 5); // 가우시안 블러의 결과 영상은 dst1에 저장 
	// 가우시안 블러의 시그마 값은 5로 설정하고 필터의 사이즈는 자동으로 결정되게 함

	tm.stop(); 
	cout << "Gaussian: " << tm.getTimeMilli() << endl; // 가우시안 블러 연산 시간

	tm.reset();
	tm.start();

	Mat dst2;
	bilateralFilter(src, dst2, -1, 10, 5); // 양방향 필터의 결과 영상은 dst2에 저장
	// 시그마 space의 값을 5로 설정하여 가우시안 블러에서 사용되는 값과 동일하게 줌
	// d 값을 -1로 결정하면, 시그마 값을 이용하여 자동으로 필터 사이즈를 결정하므로 위 가우시안 블러의 필터 사이즈와 같음

	tm.stop();
	cout << "Bilateral: " << tm.getTimeMilli() << endl; // 가우시안 블러 연산 시간

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	waitKey();
}