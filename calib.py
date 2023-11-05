import numpy as np
from matplotlib import pyplot as plt
import cv2

# Define the number of inner corners in the target
num_corners_x = 41
num_corners_y = 41

# Create arrays to store object points and image points from all images
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# Generate object points for the target
objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)

# Read images and find corners
images = ['calib_row\\mask1m5.bmp', 'calib_row\\mask10.bmp', 'calib_row\\mask15.bmp']  # Replace with your own image paths

for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    params = cv2.SimpleBlobDetector_Params()
    params.maxArea = 100000
    detector = cv2.SimpleBlobDetector_create(params)
    ret, corners = cv2.findCirclesGrid(gray, (num_corners_x, num_corners_y), None, )
    #ret, corners = cv2.findCirclesGrid(gray, (num_corners_x, num_corners_y), None, blobDetector=detector)

    # If corners are found, add object points and image points
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners
    cv2.drawChessboardCorners(img, (num_corners_x, num_corners_y), corners, ret)
    cv2.imshow('Chessboard Corners', img)
    cv2.waitKey(0)  # Adjust the delay as needed

cv2.destroyAllWindows()


# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

calibList = [ret, mtx, dist, rvecs, tvecs]
calibTxtList = [str(x) for x in calibList]

with open("calib0.txt", "w") as output:
    for row in calibTxtList:
        output.write(str(row) + '\n')

# Print the camera calibration matrix
print("Camera Calibration Matrix:")
print(mtx)
print("rvecs:")
print(rvecs)
print("tvecs:")
print(tvecs)