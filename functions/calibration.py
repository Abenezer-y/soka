import numpy as np
import cv2
import glob



# Path to the images
# images_path =f'{folder_path}*.jpg'

# Chessboard parameters
chessboard_size = (8, 6)  # Inner corners of the chessboard
square_size = 2.23  # Size of one square in any unit (could be cm or any other unit)


def calibrate(images_path, chessboard_size, square_size, filename):
    # Arrays to store object points and image points from all the images
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size


    # Loop through each image
    # images = glob.glob(images_path)
    for image_path in images_path:
        # Read the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
            # cv2.imshow('Chessboard Corners', image)
            # cv2.waitKey(500)  # Adjust the delay as needed

    # cv2.destroyAllWindows()

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Print the calibration results
    print("Calibration matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)

    # Save the calibration data
    np.savez(f'{filename}.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

    return camera_matrix, dist_coeffs



def undistort_frame(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the undistorted frame
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    return undistorted_frame