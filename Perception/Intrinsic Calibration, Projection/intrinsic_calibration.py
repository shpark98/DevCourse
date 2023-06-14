import cv2
import glob
import numpy as np
import time

DISPLAY_IMAGE = False

# # Get Image Path List
image_path_list = glob.glob("images/*.jpg")

# # Chessboard Config
BOARD_WIDTH = 9
BOARD_HEIGHT = 6
SQUARE_SIZE = 0.025

# Window-name Config
window_name = "Intrinsic Calibration"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Calibration Config
flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_FAST_CHECK
)

pattern_size = (BOARD_WIDTH, BOARD_HEIGHT)
counter = 0

image_points = list()

for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # OpneCV Color Space -> BGR
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(image_gray, pattern_size, flags)
    if ret == True:
        # print(corners)
        if DISPLAY_IMAGE:
            image_draw = cv2.drawChessboardCorners(image, pattern_size, corners, ret)
            for corner in corners:
                counter_text = str(counter)
                # print(corner)
                point = (int(corner[0][0]), int(corner[0][1]))
                cv2.putText(image_draw, counter_text, point, 2, 0.5, (0, 0, 255), 1)
                counter += 1

            counter = 0
            cv2.imshow(window_name, image_draw)
            cv2.waitKey(0)

        image_points.append(corners)
        # print(image_points)

object_points = list()
print(np.shape(image_points))

# (13, 54, 1, 2)
# (image count, featuer count, list, image_point(u, v))
# ojbect_points
# (13, 54, 1, 3)
# (image count, featuer count, list, object_point(x, y, z))

"""
 forward: Z
 right: Y
 down: X
"""

BOARD_WIDTH = 9
BOARD_HEIGHT = 6

'''
index 0 -> 0, 0, 0
index 1 -> 0, 0.025, 0
'''

for i in range(len(image_path_list)):
    object_point = list()
    height = 0
    for _ in range(BOARD_HEIGHT):
        # Loop Width -> 9
        width = 0
        for _ in range(BOARD_WIDTH):
            # Loop Height -> 6
            point = [[height, width, 0]]
            object_point.append(point)
            width += SQUARE_SIZE
        height += SQUARE_SIZE
    object_points.append(object_point)


# print(type(object_points))
object_points = np.asarray(object_points, dtype=np.float32) # list형의 object_points를 float32 데이터 타입의 Numpy 배열로 변환함
# print(type(object_points))

image_shape = np.shape(image)
image_height, image_width = image_shape[0], image_shape[1]

image_size = (image_width, image_height)
print(image_size)

# projection에 대한 정보 (카메라 매트릭스, 왜곡계수, r벡터, t벡터)
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

print("=" * 40)
print(f"re-projection error\n {ret}\n") # rojection에 대한 정보를 갖고 다시 projection을 시켰을 때 이미지를 복원한 결과이므로 re-projection이라고 함
print(f"camera matrix\n {cameraMatrix}\n")
print(f"distortion coefficientes error\n {distCoeffs}\n")
print(f"extrinsic for each image\n {len(rvecs)} {len(tvecs)}")
print("=" * 40)

for rvec, tvec, op, ip in zip(rvecs, tvecs, object_points, image_points):
    imagePoints, jacobian = cv2.projectPoints(op, rvec, tvec, cameraMatrix, distCoeffs)

    for det, proj in zip(ip, imagePoints):
        print(det, proj) # det는 코너를 detect 했을 때 이미지 포인트, proj은 projection 했을 때 이미지 포인트
        # intrinsic, extrinsic이 정확하게 calibration이 되었을 때 두 값은 동일해야 함
        # 2개의 값의 차이를 전체적으로 고려한 것이 re-projection error

start_time = time.process_time()
for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_undist = cv2.undistort(image, cameraMatrix, distCoeffs, None) # cv2.undistort() 를 사용하여 이미지를 왜곡 보정
end_time = time.process_time()
print(end_time - start_time)

start_time = time.process_time()
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, None, image_size, cv2.CV_32FC1)
for image_path in image_path_list:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_undist = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
end_time = time.process_time()
print(end_time - start_time)
