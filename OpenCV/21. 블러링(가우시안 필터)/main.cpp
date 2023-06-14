#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	imshow("src", src);

	Mat dst;
	for (int sigma = 1; sigma <= 5; sigma++) { // 시그마 값을 1부터 5까지 증가시키면서
		TickMeter tm;
		tm.start();

		GaussianBlur(src, dst, Size(0, 0), (double)sigma); // 가우시안 블러를 수행함
		
		tm.stop();
		cout << "sigma: " << sigma << ", time: " << tm.getTimeMilli() << " ms." << endl; // 가우시안 블러가 수행되는 시간을 측정해서 컨솔창에 출력함

		String desc = format("Sigma = %d", sigma);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA); // 시그마의 값을 문자열로 출력해 영상에 보여줌

		imshow("dst", dst);
		waitKey();
	}
}