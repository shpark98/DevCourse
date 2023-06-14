#include <iostream>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "mycanny.h"

using namespace std;
using namespace cv;

void myCanny(const Mat& src, Mat& dst, double threshold1, double threshold2)
{
	// 1. 가우시안 블러
	Mat gauss;
	GaussianBlur(src, gauss, Size(), 0.5);

	// 2. 소벨함수를 이용해 x축방향, y축방향으로 미분
	Mat dx, dy;
	Sobel(gauss, dx, CV_32F, 1, 0);
	Sobel(gauss, dy, CV_32F, 0, 1);

	Mat mag = Mat::zeros(src.rows, src.cols, CV_32F);
	Mat ang = Mat::zeros(src.rows, src.cols, CV_32F);
	
	for (int y = 0; y < src.rows; y++) {
		float* pDx = dx.ptr<float>(y);
		float* pDy = dy.ptr<float>(y);
		float* pMag = mag.ptr<float>(y);
		float* pAng = ang.ptr<float>(y);

		for (int x = 0; x < src.cols; x++) {
			// mag는 그래디언트의 크기
			pMag[x] = sqrt(pDx[x] * pDx[x] + pDy[x] * pDy[x]);

			// ang는 그래디언트의 방향 (값이 가장 급격하게 변하는 방향)
			if (pDx[x] == 0)
				pAng[x] = 90.f;
			else
				pAng[x] = float(atan(pDy[x] / pDx[x]) * 180 / CV_PI); // arctan를 이용해 각도를 구함
		}
	}

	// 3. Non-maximum suppression
	enum DISTRICT { AREA0 = 0, AREA45, AREA90, AREA135, NOAREA };
	const int ang_array[] = { AREA0, AREA45, AREA45, AREA90, AREA90, AREA135, AREA135, AREA0 };

	const uchar STRONG_EDGE = 255;
	const uchar WEAK_EDGE = 128;

	vector<Point> strong_edges;
	dst = Mat::zeros(src.rows, src.cols, CV_8U);

	for (int y = 1; y < src.rows - 1; y++) {
		for (int x = 1; x < src.cols - 1; x++) {
			// 그래디언트 크기가 th_low보다 큰 픽셀에 대해서만 국지적 최대 검사.
			// 국지적 최대인 픽셀에 대해서만 강한 엣지 또는 약한 엣지로 설정.
			float mag_value = mag.at<float>(y, x);
			if (mag_value > threshold1) {
				// 그래디언트 방향에 90도를 더하여 엣지의 방향을 계산 (4개 구역)
				int ang_idx = cvFloor((ang.at<float>(y, x) + 90) / 22.5f);

				// 국지적 최대 검사
				bool local_max = false;
				switch (ang_array[ang_idx]) {
				case AREA0:
					if ((mag_value >= mag.at<float>(y - 1, x)) && (mag_value >= mag.at<float>(y + 1, x))) {
						local_max = true;
					}
					break;
				case AREA45:
					if ((mag_value >= mag.at<float>(y - 1, x + 1)) && (mag_value >= mag.at<float>(y + 1, x - 1))) {
						local_max = true;
					}
					break;
				case AREA90:
					if ((mag_value >= mag.at<float>(y, x - 1)) && (mag_value >= mag.at<float>(y, x + 1))) {
						local_max = true;
					}
					break;
				case AREA135:
				default:
					if ((mag_value >= mag.at<float>(y - 1, x - 1)) && (mag_value >= mag.at<float>(y + 1, x + 1))) {
						local_max = true;
					}
					break;
				}

				// 강한 엣지와 약한 엣지 구분.
				if (local_max) {
					if (mag_value > threshold2) {
						dst.at<uchar>(y, x) = STRONG_EDGE;
						strong_edges.push_back(Point(x, y));
					} else {
						dst.at<uchar>(y, x) = WEAK_EDGE;
					}
				}
			}
		}
	}

#define CHECK_WEAK_EDGE(x, y) \
	if (dst.at<uchar>(y, x) == WEAK_EDGE) { \
		dst.at<uchar>(y, x) = STRONG_EDGE; \
		strong_edges.push_back(Point(x, y)); \
	}

	// 4. Hysterisis edge tracking
	while (!strong_edges.empty()) {
		Point p = strong_edges.back();
		strong_edges.pop_back();

		// 강한 엣지 주변의 약한 엣지는 최종 엣지(강한 엣지)로 설정
		CHECK_WEAK_EDGE(p.x + 1, p.y)
		CHECK_WEAK_EDGE(p.x + 1, p.y + 1)
		CHECK_WEAK_EDGE(p.x, p.y + 1)
		CHECK_WEAK_EDGE(p.x - 1, p.y + 1)
		CHECK_WEAK_EDGE(p.x - 1, p.y)
		CHECK_WEAK_EDGE(p.x - 1, p.y - 1)
		CHECK_WEAK_EDGE(p.x, p.y - 1)
		CHECK_WEAK_EDGE(p.x + 1, p.y - 1)
	}

	// 끝까지 약한 엣지로 남아있는 픽셀은 모두 엣지가 아닌 것으로 판단.
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			if (dst.at<uchar>(y, x) == WEAK_EDGE)
				dst.at<uchar>(y, x) = 0;
		}
	}
}