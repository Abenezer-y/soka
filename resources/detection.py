# Import libraries
import numpy as np
import pandas as pd
import streamlit as st


import cv2
import skimage
from PIL import Image, ImageColor
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error

import os
import json
import yaml
import time



def player_detection():
    t2col1, t2col2 = st.columns([1,1])
    with t2col1:
        st.header('Model Settings')
        sideref_model_conf_thresh = st.slider('Players Detection Threshold', min_value=0.0, max_value=1.0, value=0.7)
        ref_model_conf_thresh = st.slider('Goalkeepers Detection Threshold', min_value=0.0, max_value=1.0, value=0.6)
        goalkeepers_conf_thresh = st.slider('Referee Detection Threshold', min_value=0.0, max_value=1.0, value=0.5)
        player_model_conf_thresh = st.slider('Side Referee Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.5)
        keypoints_model_conf_thresh = st.slider('Ball Detection Threshold', min_value=0.0, max_value=1.0, value=0.1)
        keypoints_displacement_mean_tol = st.slider('Over all Model confidence', min_value=0.0, max_value=1.0, value=0.1,
                                                        help="Indicates the maximum allowed average distance between the position of the field keypoints\
                                                        in current and previous detections. It is used to determine wether to update homography matrix or not. ")
        detection_hyper_params = {
            0: player_model_conf_thresh,
            1: keypoints_model_conf_thresh,
            2: keypoints_displacement_mean_tol
        }
    with t2col2:

        # equalization = st.checkbox(label='Apply Hist-Equalization', value=False, key='equalization')
        st.header('Image Preprocessing')
        equalization = st.checkbox(label='Apply Hist-Equalization', value=False, key='equalization')
        noise = st.checkbox(label='Apply Noise Reduction', value=False, key='noise')
        # num_pal_colors = st.slider(label="Number of palette colors", min_value=1, max_value=5, step=1, value=3,
        #                         help="How many colors to extract form detected players bounding-boxes? It is used for team prediction.")
        # # st.markdown("---")
        # save_output = st.checkbox(label='Save output', value=False)
        # if save_output:
        #     output_file_name = st.text_input(label='File Name (Optional)', placeholder='Enter output video file name.')
        # else:
        #     output_file_name = None
    st.markdown("---")

    
    bcol1, bcol2 = st.columns([1,1])
    with bcol1:
        st.header('Tracker Settings')
        nbr_frames_no_ball_thresh = st.number_input("Ball track reset threshold (frames)", min_value=1, max_value=10000,
                                                    value=30, help="After how many frames with no ball detection, should the track be reset?")
        ball_track_dist_thresh = st.number_input("Ball track distance threshold (pixels)", min_value=1, max_value=1280,
                                                    value=100, help="Maximum allowed distance between two consecutive balls detection to keep the current track.")
        max_track_length = st.number_input("Maximum ball track length (Nbr. detections)", min_value=1, max_value=1000,
                                                    value=35, help="Maximum total number of ball detections to keep in tracking history")
        ball_track_hyperparams = {
            0: nbr_frames_no_ball_thresh,
            1: ball_track_dist_thresh,
            2: max_track_length
        }
    with bcol2:
        st.header('Output Options')
        bcol21t, bcol22t = st.columns([1,1])
        with bcol21t:
            show_k = st.toggle(label="Save video with annotation", value=False)
            show_p = st.toggle(label="Create Tactical Map Video", value=True)
        with bcol22t:
            show_pal = st.toggle(label="Develop Team Heat Map", value=True)
            show_b = st.toggle(label="Show Ball Tracks", value=True)
        plot_hyperparams = {
            0: show_k,
            1: show_pal,
            2: show_b,
            3: show_p
        }
        st.markdown("---")
        save_output = st.checkbox(label='Save output', value=False)
        if save_output:
            output_file_name = st.text_input(label='File Name (Optional)', placeholder='Enter output video file name.')
        else:
            output_file_name = None
        st.markdown('---')
        
        bcol21, bcol22, bcol23, bcol24 = st.columns([1.5,1,1,1])
        with bcol21:
            st.write('')
        with bcol22:
            # ready = True if (team1_name == '') or (team2_name == '') else False
            # start_detection = st.button(label='Start Detection', disabled=ready)
            start_detection = st.button(label='Start Detection')
        with bcol23:
            stop_btn_state = True if not start_detection else False
            stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state)
        with bcol24:
            st.write('')


    # stframe = st.empty()
    # cap = cv2.VideoCapture(tempf.name)
    # status = False

    # if start_detection and not stop_detection:
    #     st.toast(f'Detection Started!')
    #     status = detect(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
    #                      detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
    #                        num_pal_colors, colors_dic, color_list_lab)
    # else:
    #     try:
    #         # Release the video capture object and close the display window
    #         cap.release()
    #     except:
    #         pass
    # if status:
    #     st.toast(f'Detection Completed!')
    #     cap.release()