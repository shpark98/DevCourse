#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat calcGrayHist(const Mat& img) // 그레이스케일 영상 1개를 인자로 받음

// 매번 calcHist의 함수는 범용성을 위해 상당히 복잡한 형태의 인자를 받도록 구성되어있는데 단순히 그레이스케일 영상 한장으로부터 히스토그램을 계산할 때 calcHist함수의 인자를 매번 코드로 작성하는 것은 번거로움 따라서 그레이스케일 한정으로 calcHist를 랩핑하는 별도의 함수를 따로 만들어 사용하는 것이 편리함

{
	CV_Assert(img.type() == CV_8U);

	Mat hist;
	int channels[] = { 0 };
	int dims = 1;
	const int histSize[] = { 256 };
	float graylevel[] = { 0, 256 };
	const float* ranges[] = { graylevel };

	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges); 
	// 1. img 변수의 주소를 받아 
	// 2. 영상 1장을 
	// 3. 0번 채널로
    // 4. 영상 전체에 대해서 히스토그램을 계산하여
    // 5. hist라는 CV_32FC1 타입의 256 x 1 의 1차원 행렬을 만들어
    // 6. 1차원의
    // 7. hist gray 사이즈의 단계 { 256 }을 만들어 지정
    // 8. 최솟값 0, 최댓값 256 인 graylevel로 지정하고 이 배열의 이름을 인자로 받는 ranges를 지정
	
	return hist; 
	// calcHist 함수 호출 하여 hist 행렬에 값을 저장하여 hist 행렬 반환
}

Mat getGrayHistImage(const Mat& hist)
{
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	double histMax = 0.;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC1, Scalar(255)); // 세로가 100px , 가로가 256px 크기로 되어있는 imgHist의 흰색 배경을 생성한 후에
	for (int i = 0; i < 256; i++) {
		line(imgHist, Point(i, 100), // line 함수를 이용하여 히스토그램을 막대형태 그래프로 그림
			Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
	}

	return imgHist;
}

int main()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat hist = calcGrayHist(src);
	Mat imgHist = getGrayHistImage(hist);

	imshow("src", src);
	imshow("hist", imgHist);

	waitKey();
}