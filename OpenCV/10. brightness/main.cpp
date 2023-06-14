int main()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

#if 0
	Mat dst = src + 50;
	
#else
	Mat dst(src.rows, src.cols, src.type()); // src의 영상과 동일한 크기의 영상 dst 만들기

	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
		
			int v = src.at<uchar>(j,i) - 50;
			// if (v > 255) v = 255;
			// if (v < 0) v = 0;
			// v = (v > 255) ? 255 : (v < 0) ? 0 : v;
			dst.at<uchar>(j, i) = saturate_cast<uchar>(src.at<uchar>(j, i) + 50); // src의 각 픽셀에 50을 더한 값을 dst에 저장
			// 행인 j 먼저 열인 i는 두번째
			
			// 포화 연산 함수를 사용하는 이유는?
			// uchar형인 src.at<uchar>(j,i)과 int형인 50을 더하면 int 형의 변수가 나오는데 dst.at<uchar>(j,i)는 또 uchar 형임
			// 두 개의 합인 int형이 uchar 형으로 변환이 되면서 값의 변화가 생김
            // src.at<uchar>이 210 로 설정하고 2개를 더하면 260이므로 256를 넘어버려서 uchar에는 저장이 안되서 260-256 = 4 값이 저장됨
            // 따라서 기존 이미지의 밝은 부분에 대해서 포화 연산이 제대로 적용되지 않아 이상한 값이 나오는 것을 확인 할 수 있음
            // 0 보다 작아질때도 마찬가지임
            // 따라서 saturate_cast() 포화 연산 함수를 사용해 간단하게 해결 할 수 있음
		}
	}
#endif

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
}