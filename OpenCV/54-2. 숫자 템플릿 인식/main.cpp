#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat img_digits[10];

bool load_digits();
int  find_digit(const Mat& img);
void set_label(Mat& img, int idx, vector<Point>& contour);

int main()
{
	Mat src = imread("digits.png");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	if (!load_digits()) {
		cerr << "Digit image load failed!" << endl;
		return -1;
	}

	Mat src_gray, src_bin;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
//	GaussianBlur(src_gray, src_gray, Size(11, 11), 2.);
	threshold(src_gray, src_bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	// Inverse를 해서 이진화를 진행함

	vector<vector<Point>> contours;
	findContours(src_bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	// findContours를 이용해서 각각의 객체에 외곽선 좌표를 얻어낼 수 있음

	Mat dst = src.clone();

	for (unsigned i = 0; i < contours.size(); i++) { // 각각의 외곽선 객체들에 대해서
		if (contourArea(contours[i]) < 1000) // 면적이 1000보다 작으면.. (이미지의 dot를 걸러내기 위해 입력)
			continue;

		Rect rect = boundingRect(contours[i]); // 바운딩 박스 정보를 추출할 수 잇음
		int digit = find_digit(src_gray(rect)); // 부분영상을 추출
		// 추출한 부분영상을 find_digit 함수에 넣어 해당하는 숫자를 찾아 digit에 저장함

		drawContours(dst, contours, i, Scalar(0, 255, 255), 1, LINE_AA);
		set_label(dst, digit, contours[i]);
	}

//	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}

bool load_digits()
{
	for (int i = 0; i < 10; i++) {
		String filename = format("./digits/digit%d.bmp", i); 
		img_digits[i] = imread(filename, IMREAD_GRAYSCALE); // 0~9 bmp 파일을 img_digits 배열에 저장 
		if (img_digits[i].empty())
			return false;
	}

	return true;
}

int find_digit(const Mat& img) // 0 부터 9까지 어느 숫자랑 가장 비슷한지 찾는 함수
{
	int max_idx = -1;
	float max_ccoeff = -9999;

	for (int i = 0; i < 10; i++) {
		Mat src, res;
		resize(img, src, Size(100, 150)); // 부분 영상을 가로 100 세로 150으로 resize 진행
		matchTemplate(src, img_digits[i], res, TM_CCOEFF_NORMED);
		// 부분 영상을 img_digits 배열에 들어있는 10개의 template 영상과 matchTemplate를 10번 진행하고 결과를 res 행렬에 넣음

		float ccoeff = res.at<float>(0, 0); // res 행렬에 들어있는 하나의 element를 불러옴

		if (ccoeff > max_ccoeff) { 
			max_idx = i; // 이 값이 최대인 위치에 대해서 max_idx에 저장함
			max_ccoeff = ccoeff;
		}
	}

	return max_idx; // 최대인 인덱스 값을 반환함
}

void set_label(Mat& img, int digit, vector<Point>& contour)
{
	int fontface = FONT_HERSHEY_SIMPLEX;
	double scale = 0.8;
	int thickness = 1;
	int baseline = 0;

	String label = format("%d", digit);

	Size text = getTextSize(label, fontface, scale, thickness, &baseline);
	Rect r = boundingRect(contour);

	Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	rectangle(img, pt + Point(0, 2), pt + Point(text.width, -text.height), Scalar(255, 255, 255), -1);
	putText(img, label, pt, fontface, scale, Scalar(0, 0, 0), thickness, LINE_AA);
}