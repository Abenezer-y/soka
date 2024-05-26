import streamlit as st
# import cv2
import numpy as np
import pandas as pd
import tempfile
from functions.filters import *
# from app.functions.annotation.annotator import annotate_frame, annotate

from database.db import read_table, save_moment, update_detections, save_corners, Corners
# from database import db
import os
from PIL import Image
from functions.video_preprocess import video_meta, trim_video
from functions.calibration import calibrate, undistort_frame
from resources.game_report import event_report
from resources.calibrate import show_image_album
from resources.remove_distortion import remove_distortion
# from resources.tactical_map import tactical_map
# from resources.player_locations import detection
from resources.match_info import match_info
from resources.widegts import video_uploader, return_frame
from resources.events import event_page
from resources.detection import player_detection
# from resources.tab_2 import game_page
from resources.app_functions import create_or_check_folder


st.set_page_config( page_title="Soka Stats - Preprocessor", layout="wide")


def preprocessor(project_folder):
    preprocessor = st.container()
    tab1, tab2, tab3  = preprocessor.tabs(["Extract Frames", "Camera Calibration", "Removing Distortion"])

    with tab1:
  
        frame_nbr = st.slider(label="Select frame", min_value=1, max_value=video_meta_0[3], step=1, help="Select frame to analyze", key='frame0')
        frame = return_frame(tempf.name, frame_nbr)
        cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
        with cp1:
            folder_path = st.selectbox("Select a folder to save processed image:", os.listdir('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/'), key='framoe')

        with cp2:
            file_name = st.text_input("Enter file name (with extension):", "processed_image.jpg")

        with cp4:
            if st.button("Save Processed Image"):
                if folder_path:
                    if file_name is not None:
                        par_dir = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/'
                        folder_path = os.path.join(par_dir, folder_path)
                        file_path = os.path.join(folder_path, file_name)
                        # cv2.imwrite(file_path, frame)
                        st.success(f"Image saved to: {file_path}")
            
        extracted_frame = st.empty()
        extracted_frame.image(frame, use_column_width=True, channels="BGR",  caption="Processed Image",)
        st.markdown('---')

    with tab2:
        show_image_album(calibration_data)

    with tab3:
        remove_distortion(tempf)


def analyzer(project_folder, match_df):
    
    analyzer = st.container()
    tab4, tab5, tab6, tab7, tab8  = analyzer.tabs(["Match Information", "Event Collection", "Player Detection", "Tactical Map Development", "Game Report"])
    with tab4:
        match_info(input_vide_file, tempf, video_meta_0, match_df)
    with tab5:
        event_page(input_vide_file.name, tempf.name, input_vide_file, tempf, video_meta_0, project_folder)
    with tab6:
        player_detection()
    # with tab7:
        # tactical_map(input_vide_file, tempf, video_meta_0)
    with tab8:
        event_report()
            
st.sidebar.title("Main Settings")
demo_selected = st.sidebar.radio(label="Select Video Type", options=["Preprocess", "Analyze"], horizontal=True)
parent_directory = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/projects'

if demo_selected == 'Preprocess':
    # calibration_data = np.load('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/wide_angle_calib_02/calib_matrix_02.npz')
   
    input_vide_file, tempf, video_meta_0, df = video_uploader('new')
    
    if input_vide_file:
        project_folder = create_or_check_folder(parent_directory, input_vide_file.name)
        preprocessor(project_folder)
    
elif demo_selected == 'Analyze':
    calibration_data = np.load('functions/sony_4k_calibration_data.npz')
    input_vide_file, tempf, video_meta_0, df = video_uploader('new_uu')
    
    if input_vide_file:
        project_folder = create_or_check_folder(parent_directory, input_vide_file.name)
        analyzer(project_folder, df)

# if input_vide_file:
#     tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8  = st.tabs(["Extract Frames", "Camera Calibration", "Removing Distortion", "Match Information", "Tactical Map Development", "Player Detection",  "Event Collection", "Game Report"])
    
#     with tab1:
  
#         frame_nbr = st.slider(label="Select frame", min_value=1, max_value=video_meta_0[3], step=1, help="Select frame to analyze", key='frame0')
        
#         frame = return_frame(tempf.name, frame_nbr)
#         cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
#         with cp1:
#             folder_path = st.selectbox("Select a folder to save processed image:", os.listdir('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/'), key='framoe')

#         with cp2:
#             file_name = st.text_input("Enter file name (with extension):", "processed_image.jpg")

#         with cp4:
#             if st.button("Save Processed Image"):
#                 if folder_path:
#                     if file_name is not None:
#                         par_dir = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/'
#                         folder_path = os.path.join(par_dir, folder_path)
#                         file_path = os.path.join(folder_path, file_name)
#                         cv2.imwrite(file_path, frame)
#                         st.success(f"Image saved to: {file_path}")
            
#         extracted_frame = st.empty()
#         extracted_frame.image(frame, use_column_width=True, channels="BGR",  caption="Processed Image",)
#         st.markdown('---')

#     with tab2:
#         # Example usage
      
#         show_image_album(calibration_data)

#     with tab3:

    
#         remove_distortion(tempf)

#     with tab4:
     
       
#         match_info(input_vide_file, tempf, video_meta_0)

#     with tab5:
#     #     st.session_state.selected_page = 4
#         # if processed_vid:
#         tactical_map(input_vide_file, tempf, video_meta_0)

#     with tab6:
#         player_detection()
#     with tab7:
#         home_page(input_vide_file.name, tempf.name, input_vide_file, tempf, video_meta_0)
#     with tab8:
       
#         event_report()
            



            
