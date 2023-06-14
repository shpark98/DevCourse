#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("rice.png", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	// 지역 이진화
	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
	int bw = src.cols / 4;
	int bh = src.rows / 4;

	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			Mat src_ = src(Rect(x*bw, y*bh, bw, bh));
			Mat dst_ = dst(Rect(x*bw, y*bh, bw, bh));
			threshold(src_, dst_, 0, 255, THRESH_BINARY | THRESH_OTSU);
		}
	}

	// 레이블링 함수를 이용한 흰색 객체 갯수 구하기
	Mat labels;
	
	int cnt1 = connectedComponents(dst, labels); // dst의 흰색 객체의 개수에 1을 더한 값을 반환함
	cout << "# of objects in dst: " << cnt1 - 1 << endl;

	Mat dst2;
	morphologyEx(dst, dst2, MORPH_OPEN, Mat()); 
	// 지역 이진화 결과인 dst에 대해서 모폴리지 연산의 열기 연산을 수행하고 그 결과를 dst2에 저장
	// erode(dst, dst2, Mat());
	// dilate(dst2, dst2, Mat()); // 다음 2줄을 morphologyEx 로 대체해도 같은 결과가 나옴
	
	int cnt2 = connectedComponents(dst2, labels); // dst2의 흰색 객체의 개수에 1을 더한 값을 반환함
	cout << "# of objects in dst2: " << cnt2 - 1 << endl;

	imshow("src", src);
	imshow("dst", dst);
	imshow("dst2", dst2);
	waitKey();
}