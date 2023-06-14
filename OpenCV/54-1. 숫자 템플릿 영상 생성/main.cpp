#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("digit_consolas.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat src_bin;
	threshold(src, src_bin, 128, 255, THRESH_BINARY_INV); // THRESH_BINARY_INV를 진행

	Mat labels, stats, centroids;
	int label_cnt = connectedComponentsWithStats(src_bin, labels, stats, centroids);
	// 각각의 바운딩 박스를 stats 행렬로 받아옴
	
	for (int i = 1; i < label_cnt; i++) {
		int sx = stats.at<int>(i, 0);
		int sy = stats.at<int>(i, 1);
		int w = stats.at<int>(i, 2);
		int h = stats.at<int>(i, 3);
		// 바운딩 박스의 정보를 받아옴

		Mat digit;
		resize(src(Rect(sx, sy, w, h)), digit, Size(100, 150)); 
		// 부분영상을 추출하고 가로 100, 세로 150으로 resize 한 후 digit에 저장함
		String filename = cv::format("temp%d.bmp", i - 1); // filename을 temp0.bmp부터 temp9.bmp까지로 작성 
		imwrite(filename, digit); // 위에 지정한 filename으로 저장
		cout << filename << " file is generated!" << endl;
		//imshow("digit", );
		//waitKey();
	}
}