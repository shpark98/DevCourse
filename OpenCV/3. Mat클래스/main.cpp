#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void MatOp1();
void MatOp2();
void MatOp3();
void MatOp4();
void MatOp5();

int main()
{
//  MatOp1();
//	MatOp2();
//  MatOp3();
//	MatOp4();
	MatOp5();
}

void MatOp1()
{
	Mat img1; 	// empty matrix

	Mat img2(480, 640, CV_8UC1);		// unsigned char, 1-channel
	Mat img3(480, 640, CV_8UC3);		// unsigned char, 3-channels
	Mat img4(Size(640, 480), CV_8UC3);	// Size(width, height)

	Mat img5(480, 640, CV_8UC1, Scalar(128));		// 초기값을 지정 128
	Mat img6(480, 640, CV_8UC3, Scalar(0, 0, 255));	// 초기값을 지정 red

	Mat mat1 = Mat::zeros(3, 3, CV_32SC1);	// 0's matrix
	Mat mat2 = Mat::ones(3, 3, CV_32FC1);	// 1's matrix
	Mat mat3 = Mat::eye(3, 3, CV_32FC1);	// identity matrix

	float data[] = {1, 2, 3, 4, 5, 6};
	Mat mat4(2, 3, CV_32FC1, data);

	Mat mat5 = (Mat_<float>(2, 3) << 1, 2, 3, 4, 5, 6);
	Mat mat6 = Mat_<uchar>({2, 3}, {1, 2, 3, 4, 5, 6});

	mat4.create(256, 256, CV_8UC3);	// uchar, 3-channels
	mat5.create(4, 4, CV_32FC1);	// float, 1-channel

	mat4 = Scalar(255, 0, 0);
	mat5.setTo(1.f);
}

void MatOp2()
{
	Mat img1 = imread("dog.bmp");

	Mat img2 = img1;
	Mat img3;
	img3 = img1;
	// img1, img2, img3 은 같은 영상을 같게 됨(참조 형태)

	Mat img4 = img1.clone(); // img1의 복사본을 만들어 img4에 대입하는 것이므로 오른쪽(img4)에서 왼쪽(img1)으로 복사가 됨
	Mat img5; 
	img1.copyTo(img5); // img1의 img5에 복사하는 것이므로 왼쪽(img1)에서 오른쪽(img5)으로 복사가 됨
	// img4, img5 모두 같은 영상을 갖게 됨(깊은 복사 형태)

	img1.setTo(Scalar(0, 255, 255));	// yellow로 바꾸는데 img1, img2, img3가 노랑색으로 바뀌는 것을 확인 할 수 있음
	// 참조(얕은 복사) 형태로 복사하면 원본이 변경되면 변경되지만, 깊은 복사 형태로 복사하면 변경되지 않음
	
	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);
	imshow("img4", img4);
	imshow("img5", img5);

	waitKey();
	destroyAllWindows();
}

void MatOp3()
{
	Mat img1 = imread("cat.bmp");

	if (img1.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat img2 = img1(Rect(220, 120, 340, 240));
	Mat img3 = img1(Rect(220, 120, 340, 240)).clone();

	img2 = ~img2; // not 연산을 수행하여 영상을 반전함
	// img2의 영상이 반전되면서 img1에서 참조한 부분도 반전됨
	// img3는 clone을 통해 깊은 복사를 하였으므로 img2에서 영상이 바뀌어도 적용되지 않음
    
	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);

	waitKey();
	destroyAllWindows();
}

void MatOp4()
{
	Mat mat1 = Mat::zeros(3, 4, CV_8UC1); // 3행 4열의 uchar 타입의 행렬을 만들어 mat1 객체에 선언

	for (int y = 0; y < mat1.rows; y++) {
		for (int x = 0; x < mat1.cols; x++) {
			mat1.at<uchar>(y, x)++;
		}
	}
// at 함수를 이용해 모든 원소의 값을 1씩 증가시킴

	for (int y = 0; y < mat1.rows; y++) {
		uchar* p = mat1.ptr<uchar>(y);

		for (int x = 0; x < mat1.cols; x++) {
			p[x]++;
		}
	}
// ptr 함수를 이용해 모든 원소의 값을 1씩 증가시킴

	for (MatIterator_<uchar> it = mat1.begin<uchar>(); it != mat1.end<uchar>(); ++it) {
		(*it)++;
	}
// MatIterator_ 이용해 모든 원소의 값을 1씩 증가시킴

	cout << "mat1:\n" << mat1 << endl; // mat1의 모든 원소를 출력함
}

void MatOp5()
{
	float data[] = {1, 1, 2, 3};
	Mat mat1(2, 2, CV_32FC1, data);
	cout << "mat1:\n" << mat1 << endl;

	Mat mat2 = mat1.inv(); // 역행렬을 구함
	cout << "mat2:\n" << mat2 << endl;

	cout << "mat1.t():\n" << mat1.t() << endl; // transpose
	cout << "mat1 + 3:\n" << mat1 + 3 << endl; // 모든 원소에 숫자 3을 더함
	cout << "mat1 + mat2:\n" << mat1 + mat2 << endl; 
	cout << "mat1 * mat2:\n" << mat1 * mat2 << endl; // 자기 자신과 자기 자신의 역행렬을 곱하여 단위행렬 결과가 나옴
}

