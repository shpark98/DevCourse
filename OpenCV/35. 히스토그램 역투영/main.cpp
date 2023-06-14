#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	// Calculate CrCb histogram from a reference image

	Mat src = imread("cropland.png", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Rect rc = selectROI(src); // 마우스를 이용해서 특정 영역을 선택후 스페이스나 엔터키를 누르면 사용자가 선택한 영역이 rc로 전달 

	Mat src_ycrcb;
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb); // src를 ycrcb로 변환한 것을 src_ycrcb로

	Mat crop = src_ycrcb(rc); // src_ycrcb의 ROI 지정한 rc에 해당하는 부분을 crop에 저장

	Mat hist;
	int channels[] = {1, 2};
	int cr_bins = 128; int cb_bins = 128;
	int histSize[] = {cr_bins, cb_bins};
	float cr_range[] = {0, 256};
	float cb_range[] = {0, 256};
	const float* ranges[] = {cr_range, cb_range};

	// 부분 영상에 대한 히스토그램 계산
	calcHist(&crop, 1, channels, Mat(), hist, 2, histSize, ranges);
	
	// 전체 영상에 대해 히스토그램 역투영
	Mat backproj;
	calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges);

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	src.copyTo(dst, backproj);

	//imshow("src", src);
	imshow("dst", dst);
	waitKey();
}
