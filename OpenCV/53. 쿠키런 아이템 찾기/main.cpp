#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("cookierun.png"); // 입력 영상
	Mat tmpl = imread("item.png"); // 템플릿 영상

	if (src.empty() || tmpl.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat res, res_norm;
	matchTemplate(src, tmpl, res, TM_CCOEFF_NORMED);
	normalize(res, res_norm, 0, 255, NORM_MINMAX, CV_8UC1); // 위의 템플릿 매칭의 결과 행렬을 그레이스케일로 보여줌

	Mat local_max = res_norm > 220; // res_norm의 행렬이 220보다 큰 부분을 찾음

	Mat labels;
	int num = connectedComponents(local_max, labels); 
	// connectedComponent 함수를 이용해서 local_max에 6개의 하얀색 객체가 있음을 알기 때문에 num에는 7이라는 값이 전달됨

	Mat dst = src.clone();

	for (int i = 1; i < num; i++) { // for문을 통해 배경인 0번은 건너뛰고 1번 객체부터..
		Point max_loc;
		Mat mask = (labels == i); // labels 에서 i번 객체가 있는 부분만 하얀색으로 되는 마스크 행렬이 생성됨
		minMaxLoc(res, 0, 0, 0, &max_loc, mask); 
		// res에서 min값, max값을 찾는 것이 아닌 mask 부분이 흰색으로 된 부분에서만 최댓값 위치를 찾음
		

		cout << max_loc.x << ", " << max_loc.y << endl;

		Rect b_rect = Rect(max_loc.x, max_loc.y, tmpl.cols, tmpl.rows);
		rectangle(dst, b_rect, Scalar(0, 255, 255), 2);
	}

//	imshow("src", src);
//	imshow("templ", templ);
//	imshow("res_norm", res_norm);
	imshow("local_max", local_max);
	imshow("dst", dst);
	waitKey();
}