import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from functions.calibration import calibrate, undistort_frame

@st.cache_resource(experimental_allow_widgets=True)
def show_image_album(_calibration_data):
    st.title("Image Album")
    folder_path = st.selectbox("Select a folder", os.listdir('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/'), 2)
    par_dir = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/'
    folder_path = os.path.join(par_dir, folder_path)
    if not os.path.exists(folder_path):
        st.error("Folder not found!")
        return
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif']  # Add more if needed
    
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(tuple(image_extensions))]
    image_path = [os.path.join(folder_path, _) for _ in image_files]
    if not image_files:
        st.warning("No image files found in the selected folder.")
        return
    
    if "current_image_index" not in st.session_state:
        st.session_state.current_image_index = 0
    
    if st.session_state.current_image_index < 0:
        st.session_state.current_image_index = 0
    elif st.session_state.current_image_index >= len(image_files):
        st.session_state.current_image_index = len(image_files) - 1
    
    st.write(f"Image {st.session_state.current_image_index + 1} of {len(image_files)}:")
    # st.image(os.path.join(folder_path, image_files[st.session_state.current_image_index]), use_column_width=True)
    pic_c1, pic_c2 = st.columns([1, 1])
    with pic_c1:  
         st.image(os.path.join(folder_path, image_files[st.session_state.current_image_index]), use_column_width=True,  caption="Original Image")

    with pic_c2:  
        undistorted_frame = st.empty()


    col1, col2, col3 = st.columns(3)


    with col2:
        if st.button("Previous"):
            st.session_state.current_image_index -= 1
    with col3:
        if st.button("Next"):
            st.session_state.current_image_index += 1
            # calibration_data = np.load('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/wide_angle_calib_02/calib_matrix_02.npz')
            # calibration_data
            cam_mat = _calibration_data['camera_matrix']
            dist = _calibration_data['dist_coeffs']
            frame = cv2.imread(os.path.join(folder_path, image_files[st.session_state.current_image_index]))
            frame = undistort_frame(frame, cam_mat, dist)
            # st.image(os.path.join(folder_path, image_files[st.session_state.current_image_index]), use_column_width=True)

            undistorted_frame.image(frame, use_column_width=True, channels="BGR",  caption="Undistorted Image",)

    rcp1, rcp2, rcp3, rcp4,  rcp5, rcp6 = st.columns([1,1,1,1,1,1])

    with rcp1:
        row = st.number_input("Row Numbers", min_value=0, max_value=10, step=1, value = 6)

    with rcp2:
        col = st.number_input("Col Numbers", min_value=0, max_value=10, step=1, value = 8)

    with rcp3:
        size = st.number_input("Square Size", min_value=0.00, max_value=10.00, step= 0.01, value = 2.23)
    
    with rcp4:
        calib_file = st.text_input("Enter file name:", "calib_matrix_00")
        # with rcp4:
        # calib_file = st.text_input("Enter file name:", "calib_matrix_00")
    with rcp6:
        if st.button("Run Calibration"):
            st.write(row, col, size)
            chessboard_size = (col, row)  # Inner corners of the chessboard
            # square_size = 2.23  
            file_path = os.path.join(folder_path, calib_file)
            st.write(folder_path)
            st.write(image_path)
            cam_mat, dist = calibrate(image_path, chessboard_size, size, file_path)
            st.write(cam_mat, dist)



            frame = cv2.imread(os.path.join(folder_path, image_files[st.session_state.current_image_index]))
            frame = undistort_frame(frame, cam_mat, dist)
            # st.image(os.path.join(folder_path, image_files[st.session_state.current_image_index]), use_column_width=True)

            undistorted_frame.image(frame, use_column_width=True, channels="BGR",  caption="Undistorted Image",)
