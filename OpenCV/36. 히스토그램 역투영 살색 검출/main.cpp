#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	// Calculate CrCb histogram from a reference image

	Mat ref, ref_ycrcb, mask;
	ref = imread("ref.png", IMREAD_COLOR);
	mask = imread("mask.bmp", IMREAD_GRAYSCALE);
	cvtColor(ref, ref_ycrcb, COLOR_BGR2YCrCb);

	Mat hist;
	int channels[] = { 1, 2 };
	int cr_bins = 128; int cb_bins = 128; // cr, cb를 원래의 255이 아닌 128개로 나눈 것으로 설정
	int histSize[] = { cr_bins, cb_bins };
	float cr_range[] = { 0, 256 };
	float cb_range[] = { 0, 256 };
	const float* ranges[] = { cr_range, cb_range };

	calcHist(&ref_ycrcb, 1, channels, mask, hist, 2, histSize, ranges); // mask 이미지를 가지고 Mat 클래스 타입 hist에 저장

#if 1
	Mat hist_norm;
	normalize(hist, hist_norm, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("hist_norm", hist_norm);
#endif

	Mat src, src_ycrcb;
	src = imread("kids.png", IMREAD_COLOR);
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

	Mat backproj;
	calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges);
	
	GaussianBlur(backproj, backproj, Size(), 1.0); // 가우시안 블러딩 결과를 약간 스무딩함, 간혹 튀는 값인 노이즈를 조금 상쇄시키기 위함
	backproj = backproj > 50; // 50 보다 작은 것은 무시

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	src.copyTo(dst, backproj); // src 영상을 backproj에서 흰색으로 되어있는 부분만 dst에로 저장

	imshow("ref", ref);
	imshow("mask", mask);
	imshow("src", src);
	imshow("backproj", backproj);
	imshow("dst", dst);
	waitKey();
}