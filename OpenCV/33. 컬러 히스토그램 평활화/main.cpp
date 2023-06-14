#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("mandrill.bmp", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat src_ycrcb;
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb); // BGR 에서 YCrCb로 변경하면서 src_ycrcb는 CV_8UC3의 형태를 가짐

	vector<Mat> planes;
	split(src_ycrcb, planes); // planes안에 3개의 Mat 객체가 들어있는 형태로 변경됨 planes[0] : y, planes[1] : Cr, planes[2] : Cb

	equalizeHist(planes[0], planes[0]); // planes[0]만 평활화

	Mat dst_ycrcb;
	merge(planes, dst_ycrcb);

	Mat dst;
	cvtColor(dst_ycrcb, dst, COLOR_YCrCb2BGR);

	imshow("src", src);
	imshow("dst", dst); // imshow 함수는 무조건 BGR 색상 성분의 Mat 객체를 전달받아야함 따라서 위의 dst_ycrcb를 YCrCb 성분의 형태를 BGR로 변환해야함
	waitKey();
}