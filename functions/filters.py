import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import streamlit as st

# calibration_data = np.load('sony_4k_calibration_data.npz')
# camera_matrix = calibration_data['camera_matrix']
# dist_coeffs = calibration_data['dist_coeffs']

@st.cache_resource
def bw_filter(img):
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return img_gray

@st.cache_resource
def img_blur(img, k=5, s=5):
    img1_blur = cv2.GaussianBlur(img, (k,k), s, s)
    return img1_blur

# @st.cache_resource
# def vignette(img, level=2):
#     height, width = img.shape[:2]

#     # Generate vignette mask using Gaussian kernels.
#     X_resultant_kernel = cv2.getGaussianKernel(width, width / level)
#     Y_resultant_kernel = cv2.getGaussianKernel(height, height / level)

#     # Generating resultant_kernel matrix.
#     kernel = Y_resultant_kernel * X_resultant_kernel.T
#     mask = kernel / kernel.max()

#     img_vignette = np.copy(img)

#     # Apply the mask to each channel in the input image.
#     for i in range(3):
#         img_vignette[:, :, i] = img_vignette[:, :, i] * mask

#     return img_vignette

# @st.cache_resource
# def sepia(img):
#         img_sepia = img.copy()
#         # Converting to RGB as sepia matrix below is for RGB.
#         img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
#         img_sepia = np.array(img_sepia, dtype=np.float64)
#         img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
#                                                         [0.349, 0.686, 0.168],
#                                                         [0.272, 0.534, 0.131]]))
#         # Clip values to the range [0, 255].
#         img_sepia = np.clip(img_sepia, 0, 255)
#         img_sepia = np.array(img_sepia, dtype=np.uint8)
#         img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
#         return img_sepia

# @st.cache_resource
# def pencil_sketch(img, ksize=5):
    # img_blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
    # img_sketch, _ = cv2.pencilSketch(img_blur)
    # return img_sketch
@st.cache_resource
def equqlize_hist(image):
    img = image.copy()
    # Convert to HSV.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Perform histogram equalization only on the V channel, for value intensity.
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # Convert back to BGR format.
    img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_eq


@st.cache_resource
def thresh_img(img, blockSize=13, c=7, t_val=50, max_val = 255, t_type='Global'):
    imageGray = bw_filter(img)
    if t_type == 'Golobal':
        ret, thresh = cv2.threshold(img, t_val, max_val, cv2.THRESH_BINARY)
    else:
        thresh = cv2.adaptiveThreshold(imageGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, c)
    return thresh

@st.cache_resource
def find_shapes(img, blockSize=13, c=7):
    imageGray = bw_filter(img)
    # Apply Gaussian blur with kernel size 7x7.
    img1_blur = cv2.GaussianBlur(imageGray, (11,11), 5, 5)
    thresh = cv2.adaptiveThreshold(img1_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, c)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,0,255), 3)
    return img

@st.cache_resource
def canny_edge(img, v1=50, v2=200, blockSize=13, c=7):
    imageGray = bw_filter(img)
    # Apply Gaussian blur with kernel size 7x7.
    img1_blur = cv2.GaussianBlur(imageGray, (7,7), 7, 7)
    thresh = cv2.adaptiveThreshold(img1_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, c)
    blurred_edges  = cv2.Canny(thresh, threshold1 = v1, threshold2 = v2)
    return blurred_edges


@st.cache_resource
def region_of_interest(img, lower_array = [35, 50, 50], upper_array = [80, 255, 255]):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Perform histogram equalization only on the V channel, for value intensity.
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # Set range for green color.
    g_lb = np.array( lower_array , np.uint8)
    g_ub = np.array( upper_array, np.uint8)

    color_mask = cv2.inRange(img_hsv, g_lb, g_ub)
    img_seg = cv2.bitwise_and(img, img, mask = color_mask)

    return img_seg


@st.cache_resource
def find_field_lines(image, thresh=100, gap=50, minLen = 100):
    #  undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
     img_seg = region_of_interest(image, lower_array = [32, 50, 50], upper_array = [80, 255, 255])
     img1_blur = cv2.GaussianBlur(img_seg, (5,5), 5, 5)
     thresh = thresh_img(img1_blur, t_type='Adaptive')
     edges  = cv2.Canny(thresh, threshold1 = 50, threshold2 = 100)
    #  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, maxLineGap=50, minLineLength=20)
     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, thresh, maxLineGap=gap, minLineLength=minLen)

     for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
     return image, lines




@st.cache_resource
def draw_lines_from_cv(img, lines, color=[255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    for line in lines:
        for line in lines:           
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


@st.cache_resource
def draw_lines_from_list(img, lines, color=[255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    for line in lines:
        for line in lines:
            x1 = line[0]
            y1 = line[1] 
            x2 = line[2]
            y2 = line[3]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


@st.cache_resource
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Utility for defining Line Segments."""
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]),
        minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    # draw_lines_from_cv(line_img, lines)
    return line_img, lines


@st.cache_resource
def separate_line_orientation(lines, ag=75):
    vertical_lines = []
    horizontal_lines = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate angle of the line segment
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            # Check if line segment is close to horizontal or vertical
            if angle < -(ag) or angle > ag:  # Horizontal lines
                horizontal_lines.append([x1, y1, x2, y2])
            elif -(ag) <= angle <= (ag):  # Vertical lines
                vertical_lines.append([x1, y1, x2, y2])
            
    return vertical_lines, horizontal_lines


@st.cache_resource
def separate_left_right_lines(lines):
    """Separate left and right lines depending on the slope."""
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2: # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif y1 < y2: # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines
