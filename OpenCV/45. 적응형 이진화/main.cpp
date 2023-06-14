#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int block_size = 51;
Mat src, dst;

void on_trackbar(int, void*)
{
	int bsize = block_size; 

	if ((bsize & 0x00000001) == 0) // 짝수인지 아닌지 판단. 맨 마지막 비트가 1인지 아닌지로 판별해 짝수 판별
		bsize--; // 짝수일 경우 bsize를 1 감소

	if (bsize < 3) 
		bsize = 3; // 블럭 사이즈 값이 3보다 작아질 경우 3이 되도록 설정

	adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, bsize, 5);
	// 평균값을 계산할때 가우시안 블러 형태를 사용하는 것이 결과가 좋음
	// 일반적인 이진화를 수행하고 위에서 지정한 블럭사이즈 bsize를 사용
	// 현재 블럭에서 구한 평균값에서 5를 뺀 값을 이용해서 이진화를 수행함
	
	imshow("dst", dst);
}

int main()
{
	src = imread("sudoku.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	namedWindow("src");
	imshow("src", src);

	namedWindow("dst");
	createTrackbar("Block Size", "dst", &block_size, 201, on_trackbar);
	on_trackbar(0, 0); // Call the function to initialize

	waitKey();
}