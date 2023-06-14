#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat src_edge;
	Canny(src, src_edge, 50, 150); // 캐니 엣지 결과를 src_edge에 저장

	TickMeter tm;
	tm.start();

	vector<Vec2f> lines1;
	HoughLines(src_edge, lines1, 1, CV_PI / 180, 250); // src_edge의 검출된 직선의 정보를 lines1로 저장
	// 1 픽셀 단위로 rho 값을 변경하고 1도 간격으로 theta를 변경
	// 축적 배열에서 250보다 큰 경우에 한에서만 직선으로 검출

	tm.stop();

	cout << "HoughLines(): " << tm.getTimeMilli() << "ms." << endl; // 시간측정 결과

	Mat dst1;
	cvtColor(src_edge, dst1, COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines1.size(); i++) { // for를 이용해서 직선을 그려주는 코드
		float r = lines1[i][0], t = lines1[i][1]; // 직선의 극좌표계 r = rho, t = theta의 값
		double cos_t = cos(t), sin_t = sin(t); // cos_t는 cos theta, sin_t는 sin theta 값
		double x0 = r*cos_t, y0 = r*sin_t; // theta 각도로 rho 만큼 간 점의 좌표
		double alpha = 1000; // 알파값이 작으면 직선이 짧아지고, 크면 직선이 길어짐

		Point pt1(cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t));
		Point pt2(cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t));
		line(dst1, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
	}

	tm.reset();
	tm.start();

	vector<Vec4i> lines2;
	HoughLinesP(src_edge, lines2, 1, CV_PI / 180, 160, 50, 5); // src_edge의 직선의 성분을 찾아 lines2에 저장
	// 축적 배열에서의 threshold 값을 160으로 설정
	// 50 픽셀 이상의 선분만 구하고 5픽셀 이내로 끊어져있으면 그 선분들은 하나의 성분으로 인식

	tm.stop();

	cout << "HoughLinesP(): " << tm.getTimeMilli() << "ms." << endl;

	Mat dst2;
	cvtColor(src_edge, dst2, COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines2.size(); i++) {
		Vec4i l = lines2[i];
		line(dst2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	waitKey();
}