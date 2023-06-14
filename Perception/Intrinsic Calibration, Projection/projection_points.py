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

## Part 2 Project Points

# A. drawFrameAxes
for rvec, tvec, image_path in zip(rvecs, tvecs, image_path_list):
    # rvec -> length 3 vector(rodrigues, eular) -> 3x3 rotation matrix
    print(rvec, tvec, image_path)

    # read image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # draw frame coordinate(좌표계 그리기)
    image = cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.03, 3) 

    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)

# B. projection frame points

## object points -> [X, Y, Z]
## camera coordinate
# X -> down
# Y -> right
# Z -> forward

xy_points = list() # XY plane
yz_points = list() # YZ plane
zx_points = list() # XZ plnae

# 1cm point, meter
point_size = 0.001

for i in range(len(image_path_list)):

    # xy plane
    xy_point = list()
    x = 0
    y = 0
    z = 0
    for _ in range(100):
        y = 0
        for _ in range(100):
            point = [[x, y, z]]
            y += point_size
            xy_point.append(point)
        x += point_size
    xy_points.append(xy_point)

    # yz plane
    yz_point = list()
    x = 0
    y = 0
    z = 0
    for _ in range(100):
        z = 0
        for _ in range(100):
            point = [[x, y, z]]
            z += point_size
            yz_point.append(point)
        y += point_size
    yz_points.append(yz_point)

    # zx plane
    zx_point = list()
    x = 0
    y = 0
    z = 0
    for _ in range(100):
        x = 0
        for _ in range(100):
            point = [[x, y, z]]
            x += point_size
            zx_point.append(point)
        z += point_size
    zx_points.append(zx_point)

xy_points = np.asarray(xy_points, dtype=np.float32)
yz_points = np.asarray(yz_points, dtype=np.float32)
zx_points = np.asarray(zx_points, dtype=np.float32)

# BGR
blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)

for rvec, tvec, image_path, xy, yz, zx in zip(rvecs, tvecs, image_path_list, xy_points, yz_points, zx_points):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    def projection_points(image, object_points, rvec, tvec, cameraMatrix, distCoeffs, color) :
        image_points, jacobians = cv2.projectPoints(object_points, rvec, tvec, cameraMatrix, distCoeffs)
        print(jacobians)
        for image_point in image_points:
            image_point = image_point[0]
            x = image_point[0]
            y = image_point[1]

            # print(point)
            if x > 0 and y > 0 and x < image_width and y < image_height:
                point = (int(x), int(y))
                image = cv2.circle(image, point, 1, color)

        return image    

    image = projection_points(image, xy, rvec, tvec, cameraMatrix, distCoeffs, blue_color)
    image = projection_points(image, yz, rvec, tvec, cameraMatrix, distCoeffs, green_color)
    image = projection_points(image, zx, rvec, tvec, cameraMatrix, distCoeffs, red_color)

    image = cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.03, 3)
    
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
