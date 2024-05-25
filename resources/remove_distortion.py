import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
from functions.filters import *
from functions.calibration import calibrate, undistort_frame
import os


def write_video(video_path, width, height,  out_dir):
    if type(video_path) == str:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = video_path

    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter(out_dir, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        cam_mat = calibration_data['camera_matrix']
        dist = calibration_data['dist_coeffs']
        frame = undistort_frame(frame, cam_mat, dist)
        out.write(frame)

    cap.release()
    out.release()

@st.cache_resource(experimental_allow_widgets=True)
def remove_distortion(tempf):
        cap_temp = cv2.VideoCapture(tempf.name)
        frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_nbr = st.slider(label="Select frame", min_value=1, max_value=frame_count, step=1, help="Select frame to analyze", key='frame')
        cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
        success, frame = cap_temp.read()

        pic_col1, pic_col2 = st.columns([1, 1])
        with pic_col1:
            orginal_frame = st.empty()
            orginal_frame.image(frame, use_column_width=True, channels="BGR",  caption="Original Image",)
            st.write(frame.shape)
        with pic_col2:
            calibration_data = np.load('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/wide_angle_calib_02/calib_matrix_02.npz')
            calibration_data
            cam_mat = calibration_data['camera_matrix']
            dist = calibration_data['dist_coeffs']
            frame = undistort_frame(frame, cam_mat, dist)
            h, w, _ = frame.shape
            undistorted_frame = st.empty()
            undistorted_frame.image(frame, use_column_width=True, channels="BGR",  caption="Undistorted Image",)
            st.write(frame.shape)
        st.markdown('---')
        fcp1, fcp2, fcp3, fcp4, fcp5, fcp6, fcp7  = st.columns([1,1,1,1, 1, 1, 1])
    
        with fcp2:
            par_dir = '/Users/abenezer/Documents/Game Footage Workdir/varnero/'
            folder_path = st.selectbox("Select a folder", os.listdir(par_dir), key='kkfr')
            folder_path = os.path.join(par_dir, folder_path)

        with fcp3:
            file_name = st.text_input("Video name (extension):", "video.mp4", key='kkkdv')
        with fcp4:
            width = st.number_input("Width:", 0, 3840, w, key='kkfkdv')
        with fcp5:
            height = st.number_input("Height:", 0, 2160, h, key='kkdskdv')
        with fcp7:
            if st.button('Save Video'):
                folder_path = os.path.join(folder_path, file_name)
                try:
                    write_video(tempf.name, width, height, folder_path)
                    st.success(f"Video saved to: {folder_path}")
                except:
                    st.error('Video Not Saved!')