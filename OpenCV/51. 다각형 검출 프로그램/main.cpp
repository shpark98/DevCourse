#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void setLabel(Mat& img, const vector<Point>& pts, const String& label) 
{
	Rect rc = boundingRect(pts); // 특정 좌표들을 둘러싸고 있는 바운딩 박스를 계산
	rectangle(img, rc, Scalar(0, 0, 255), 1); // 바운딩 박스를 빨간색으로 그림을 그림
	putText(img, label, rc.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
	// rc.tl()은 top left의 약자로 사각형에서 좌측 상단의 좌표를 반환함
	// 사각형 좌측 상단에 label 텍스트를 삽입
}

int main()
{
	Mat img = imread("polygon.bmp", IMREAD_COLOR);

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY); // 그레이스케일로 변환하고 gray에 저장

	Mat bin;
	threshold(gray, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU); 
	// Otsu 방법으로 그레이스케일 영상을 INV 이진영상으로 변환하고 bin에 저장
	// 각각의 도형이 배경보다 진해서 단순하게 이진화하면 배경의 흰색, 도형이 검정색 형태로 이진화됨
	// 그렇게 되면 findContours 함수, connectedComponents에서 오동작을 함

	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); 
	// 각각의 객체 안에 Hole이 없으므로 외곽선만 검출하기 위해 RETR_EXTERNAL 설정

	for (vector<Point>& pts : contours) {
		if (contourArea(pts) < 400) // 도형의 면적이 400보다 작으면 검출 x
		// 이진화를 하게되면 외곽선 주변에서 자잘한 크기의 잡음이 발생할 수 있음. 이러한 잡음에 대해서 검출 x
			continue;

		vector<Point> approx;
		approxPolyDP(pts, approx, arcLength(pts, true)*0.02, true); 
		// 각각의 외곽선을 근사화 시킴
		// 근사화가 잘 되면, approx는 꼭지점으로 이루어진 vector<Point> 타입으로 채워짐
		
		if (!isContourConvex(approx))
			continue;

		int vtc = (int)approx.size(); // arrpox에 들어있는 점들의 개수를 파악

		if (vtc == 3) {
			setLabel(img, pts, "TRI"); // 파악한 개수가 3이면 삼각형
		} else if (vtc == 4) { 
			setLabel(img, pts, "RECT"); // 파악한 개수가 4면 사각형
		} else { // 나머지 코드에서는 원을 판별하는 코드
			double len = arcLength(pts, true);
			double area = contourArea(pts);
			double ratio = 4. * CV_PI * area / (len * len);

			if (ratio > 0.85) { // 1에 가까운 값이면 원으로 지정
				setLabel(img, pts, "CIR");
			}
		}
	}

	imshow("img", img);
	waitKey();
}