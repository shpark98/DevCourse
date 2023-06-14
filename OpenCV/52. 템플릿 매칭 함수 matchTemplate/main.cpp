#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("circuit.bmp", IMREAD_GRAYSCALE); // 입력 영상
	Mat tmpl = imread("crystal.bmp", IMREAD_GRAYSCALE); // 템플릿 영상
	// 템플릿 매칭은 반드시 그레이스케일 영상을 사용해야하는 것은 아님
	// 그레이스케일 영상을 사용했을때 디테일이 많고 메모리도 효율적으로 사용하고 연산도 빨리 하는 장점이 있음

#if 0
	src = imread("wheres_wally.jpg", IMREAD_GRAYSCALE); // IMREAD_COLOR
	tmpl = imread("wally.bmp", IMREAD_GRAYSCALE);
#endif

	if (src.empty() || tmpl.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

#if 1 // 그대로 수행하면 너무 쉽게 찾으므로 영상을 살짝 변형을 함
	src = src + 50; // src의 영상을 바꾸기 위해 밝기를 조절함

	Mat noise(src.size(), CV_32S); // 부호가 있는 정수형 형태로 잡음을 생성
	randn(noise, 0, 10); // 10, 50, 100 // 표준편차가 10인 가우시안 잡음을 발생시킴
	add(src, noise, src, noArray(), CV_8U); // 지저분한 형태의 입력 영상으로 만들고 결과 행렬은 그레이스케일 형식으로 저장하게 함. 
	// 이렇게 설정해야 음수로 설정된 Noise값도 추가가 되므로 조금 더 자연스러운 형태의 잡음이 추가가 됨
#endif

#if 1
	GaussianBlur(src, src, Size(), 1);
	GaussianBlur(tmpl, tmpl, Size(), 1);
#endif

#if 0
	resize(src, src, Size(), 0.9, 0.9); // 0.8, 0.7
#endif

#if 0
	Point2f cp(src.cols / 2.f, src.rows / 2.f);
	Mat rot = getRotationMatrix2D(cp, 10, 1); // 20, 30
	warpAffine(src, src, rot, src.size());
#endif

	Mat res, res_norm;
	matchTemplate(src, tmpl, res, TM_CCOEFF_NORMED);
	// TM_CCOEFF_NORMED로 지정했으므로 res 값은 -1 ~ 1의 값으로 나타는데 1에 가까울수록 좀 더 비슷함
	normalize(res, res_norm, 0, 255, NORM_MINMAX, CV_8U);
	// res의 값이 어떻게 구성되어있는지 그레이스케일로 변환해서 눈으로 확인해보자

	double maxv;
	Point maxloc;
	minMaxLoc(res, 0, &maxv, 0, &maxloc); // 최댓값과 최대위치를 찾기 위해
    // res_norm으로 안하는 이유는 실수형을 정수형으로 바꾸면서 운이 없으면 최댓값 위치가 여러군데서 발생할 수 있음

	cout << "maxv: " << maxv << endl;
	cout << "maxloc: " << maxloc << endl; // 최댓값, 최댓값 위치 출력

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR); // src 그레이스케일 영상을 dst 컬러영상으로 바꿈

	rectangle(dst, Rect(maxloc.x, maxloc.y, tmpl.cols, tmpl.rows), Scalar(0, 0, 255), 2); 
	// Rect 함수를 이용해서 검출된 위치를 표현
	// 그리는 위치를 사각형의 좌측상단을 지정하고 템플릿 영상의 크기와 동일하게 사각형의 크기를 지정함
	// 빨간색 2pixel의 두께로 사각형을 그림

//	imshow("src", src);
	imshow("tmpl", tmpl);
	imshow("res_norm", res_norm);
	imshow("dst", dst);
	waitKey();
}