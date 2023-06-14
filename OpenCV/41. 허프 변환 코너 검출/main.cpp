#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	// goodFeaturesToTrack

	TickMeter tm1;
	tm1.start();

	vector<Point2f> corners;
	goodFeaturesToTrack(src, corners, 400, 0.01, 10); // goodFeaturesToTrack

	tm1.stop();
	cout << "goodFeaturesToTrack() took " << tm1.getTimeMilli() << "ms." << endl;

	Mat dst1;
	cvtColor(src, dst1, COLOR_GRAY2BGR);

	for (size_t i = 0; i < corners.size(); i++) { // 코너점 위치에 원을 그림
		circle(dst1, corners[i], 5, Scalar(0, 0, 255), 2);
	}

	// FAST

	TickMeter tm2;
	tm2.start();

	vector<KeyPoint> keypoints;
	FAST(src, keypoints, 60); 	// FAST 현재 점보다 60만큼 밝거나 60만큼 어두운 픽셀이 연속으로 9개 나오면 검출
//	Ptr<FeatureDetector> detector = FastFeatureDetector::create(60);
//	detector->detect(src, keypoints);

	tm2.stop();
	cout << "FAST() took " << tm2.getTimeMilli() << "ms." << endl;

	Mat dst2;
	cvtColor(src, dst2, COLOR_GRAY2BGR);

	for (const KeyPoint& kp : keypoints) { // 코너점 위치에 원을 그림
		circle(dst2, Point(kp.pt.x, kp.pt.y), 5, Scalar(0, 0, 255), 2, LINE_AA);
	}

//	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	waitKey();
}