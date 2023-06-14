#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cerr << "Camera open failed!" << endl;
		return -1;
	}

	int  fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
	double fps = 30;
	Size sz = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT)); 
	// 현재 카메라의 설정에서 Width, Height를 가져와 그 값을 정수 값으로 변환하고 Size 객체 Sz를 생성함
	// Size sz((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT)); 로 바꿔 쓸 수 있음

	cout << "FPS = " << fps << endl;
	cout << "Size = " << sz << endl;

	VideoWriter output("output.avi", fourcc, fps, sz); // output.avi 파일로 저장

	if (!output.isOpened()) { // 동영상 파일 저장을 위해 파일 open을 실패했을 경우 에러문구와 함께 프로그램 종류 
		cerr << "output.avi open failed!" << endl;
		return -1;
	}

	int delay = cvRound(1000 / fps); 
	Mat frame, edge;
	
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		Canny(frame, edge, 50, 150);
		cvtColor(edge, edge, COLOR_GRAY2BGR); 
		// 그레이스케일 영상인 edge 파일을 BGR 파일로 변경해야 정상적으로 저장이 됨
		// 안하면 저장이 안되는 이유 : 위에서 output을 color로 만들었으므로 output을 수정하든 이렇게 변환해주든 둘 중 하나를 진행해야 저장이 가능

		output << frame; // output에 현재 frame을 저장
		output << edge; 

		imshow("frame", frame);
		imshow("edge", edge);

		if (waitKey(delay) == 27)
			break;
	}

	cout << "output.avi file is created!!!" << endl;
	
	output.release();
	cap.release();
	destroyAllWindows();
}