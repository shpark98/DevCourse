#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture cap("../../data/test_video.mp4");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return -1;
	}

	Mat frame;
	while (true) {
		cap >> frame;

		if (frame.empty()) {
			cerr << "Empty frame!" << endl;
			break;
		}

		line(frame, Point(570, 280), Point(0, 560), Scalar(255, 0, 0), 2);
		line(frame, Point(570, 280), Point(1024, 720), Scalar(255, 0, 0), 2); // 파랑색 선 2개 그리기

		int pos = cvRound(cap.get(CAP_PROP_POS_FRAMES)); // 프레임을 받아오고
		String text = format("frame number: %d", pos); // text 변수에 프레임값을 format에 맞게 넣어주고
		putText(frame, text, Point(20, 50), FONT_HERSHEY_SIMPLEX, 
			0.7, Scalar(0, 0, 255), 1, LINE_AA); // text 변수 문자열을 영상에 출력함 Point(20,50) 좌측상단으로부터 오른쪽으로 20 아래로 50 점에서 오른쪽 상단으로 문자열이 출력 됨

		imshow("frame", frame);

		if (waitKey(10) == 27)
			break;
	}

	cap.release();
	destroyAllWindows();
}