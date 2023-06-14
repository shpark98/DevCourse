#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat src;
Point ptOld;
void on_mouse(int event, int x, int y, int flags, void*);

int main(void)
{
	src = imread("lenna.bmp");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}
	
	namedWindow("src");
	setMouseCallback("src", on_mouse); // 마우스 이벤트 처리를 위한 콜백 함수

	imshow("src", src);
	waitKey();
}

void on_mouse(int event, int x, int y, int flags, void*)
{
	switch (event) { // 특정 이벤트에서만 특정 활동을 할 수 있게 switch문이나 if문을 사용
	case EVENT_LBUTTONDOWN: // 왼쪽 마우스를 눌렀을 때 해당 마우스의 좌표를 출력
		ptOld = Point(x, y); // 마우스 왼쪽 버튼이 눌렸을때 ptOld 초기화
		cout << "EVENT_LBUTTONDOWN: " << x << ", " << y << endl;
		break;
	case EVENT_LBUTTONUP: // 왼쪽 마우스를 떼었을 때 해당 마우스의 좌표를 출력
		cout << "EVENT_LBUTTONUP: " << x << ", " << y << endl;
		break;
	case EVENT_MOUSEMOVE:
		if (flags & EVENT_FLAG_LBUTTON) { 
		// flags == EVENT_FLAG_LBUTTON 보다는 flags & EVENT_FLAG_LBUTTON 로 bit가 설정되어있는지 확인하는 것이 적절한 코드 작성 방법
		// flags == EVENT_FLAG_LBUTTON 를 설정했을 때는 ctrl(MouseEventFlags 에서 8)을 누른상태에서 왼쪽 버튼을 누르면 코드가 실행되지 않음
        // 1 == 9(8+1) 가 성립되지 않으므로     
        
			//cout << "EVENT_MOUSEMOVE: " << x << ", " << y << endl;
			//circle(src, Point(x, y), 2, Scalar(0, 255, 255), -1, LINE_AA);
			
			line(src, ptOld, Point(x, y), Scalar(0, 255, 255), 3, LINE_AA);
			// 마우스가 움직이면 circle 함수보다는 line 함수를 이용해서 맨 처음 포인트부터 현재 포인트까지 선을 그어야 중간에 끊임이 없음
			
			ptOld = Point(x, y); // ptOld 현재 포인트로 다시 업데이트
			imshow("src", src);
		}
		break;
	default:
		break;
	}
}
