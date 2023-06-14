#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void sharpen_mean();
void sharpen_gaussian();

int main(void)
{
	// sharpen_mean();
	sharpen_gaussian();
}

void sharpen_mean()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE); // src에 영상 저장

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	
	/*
	float sharpen[] = { // 주변에는 -1 / 9 로 채워져있고 가운데는 17 / 9 로 구성된 3 x 3 행렬의 float형 sharpen 생성
		-1 / 9.f, -1 / 9.f, -1 / 9.f,
		-1 / 9.f, 17 / 9.f, -1 / 9.f,
		-1 / 9.f, -1 / 9.f, -1 / 9.f
	};
	
	Mat kernel(3, 3, CV_32FC1, sharpen); # sharpen을 초기값으로 갖는 3 x 3 크기의 커널을 만듬

	Mat dst;
	filter2D(src, dst, -1, kernel); # 필터 마스크 행렬을 지정하면 샤프닝이 진행됨
	*/
	
	Mat blr;
	blur(src, blr, Size(3,3));
	
	Mat dst 2 & src - blr;
	
	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void sharpen_gaussian()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat srcf;
	src.convertTo(srcf, CV_32FC1); // 입력 영상인 그레이스케일 src를 float 타입으로 변환해 srcf에 저장

	for (int sigma = 1; sigma <= 5; sigma++) {
		Mat blr;
		GaussianBlur(srcf, blr, Size(), sigma);  // 가우시안 블러 입력영상으로 srcf
		// 소수점 이하의 연산이 잘려나가지 않게 하기 위해 32F1의 src가 아닌 srcf로 진행

		float alpha = 1.0f;
		Mat dst = (1.f + alpha) * srcf - alpha * blr; // 샤프닝 된 영상 결과를 생성하고 dst는 float 타입의 실수형 행렬로 만들어짐

		dst.convertTo(dst, CV_8UC1); // float 타입으 ㅣ실수형 행렬인 dst를 화면에 보여주기 위해 다시 그레이스케일로 변환 

		String desc = format("sigma: %d", sigma); // 시그마 값 출력
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

		imshow("dst", dst);
		waitKey();
	}

	destroyAllWindows();
}