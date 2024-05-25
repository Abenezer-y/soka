import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
from PIL import Image, ImageColor
import os
from functions.calibration import calibrate, undistort_frame
from database.db import read_table, Match, Detection, update_match, save_moment, read_db, save_match, update_detections, save_corners, Corners
# from database import db
from functions.model import run_inference, run_inferene_on_frame
from streamlit_image_coordinates import streamlit_image_coordinates
from resources.widegts import return_frame, video_uploader
import skimage
from sklearn.metrics import mean_squared_error
from annotation.annotator import crop_detections
import supervision as sv
from resources.app_functions import predict_team, create_colors_info, crop_detections, get_pixel_color, grid_image


ellipse_annotator = sv.EllipseAnnotator()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)


model_path = '/Users/abenezer/Downloads/best (12).pt'

def table_form(df, name, frame, frame_nbr, place_holder):

    add_info = st.session_state["add_info"] if "add_info" in st.session_state else False
    if df.empty:
        add_info = st.checkbox('Add Match Info', value=add_info)
        if add_info:
            teamA = st.text_input('Team A', key='teamA')
            teamB = st.text_input('Team B', key='teamB')
            stad = st.text_input('Stadium', key='stadium')
            date = st.date_input('Date', key='date')
            save = st.button('-- Save -- ', key='save', )

            if save:
                save_match(name, teamA, teamB, stad, str(date))
                st.session_state["add_info"] = False




    else:
        st.write(df)
        team1_name = df['teamA'].values[0]
        team2_name = df['teamB'].values[0]
        trasform =  st.checkbox("Identify Team Colors", key='detect')
        team1_name = df['teamA'].values[0]
        team2_name = df['teamB'].values[0]
        match_id = df['id'].values[0]

        if trasform:
            with st.spinner('Detecting players in selected frame..'):
                df_detections = read_table(Detection)

                if not df_detections.empty:
                    df_detections = df_detections[(df_detections['video_id'] == match_id)&(df_detections['frame'] == 'frame_nbr')]
                if df_detections.empty:
                    image, detection, imgs_list = run_inferene_on_frame(model_path, frame, name, frame_nbr)
                else:
                    imgs_list = crop_detections(frame, df_detections)

                concat_det_imgs = grid_image(imgs_list)
                
                st.write("Detected players")

                value = streamlit_image_coordinates(concat_det_imgs, key="numpy")

                radio_options_1 =[f"{team1_name} Shirt", f"{team1_name} Short", f"{team1_name} Socks", f"{team1_name} GK"]
                radio_options_2 =[f"{team2_name} Shirt", f"{team2_name} Short", f"{team2_name} Socks", f"{team2_name} GK"]

                if value is not None:
                    picked_color = concat_det_imgs[value['y'], value['x'], :]
                    
                    st.write("Boxes below can be used to manually adjust selected colors.")
                    team_1 = st.checkbox('Team one', key='t1' )
                    
                        # st.write(team_1)
                    active_color = st.radio(label="Select which team color to pick from the image above", options=radio_options_1, horizontal=True,
                                        help="Chose team color you want to pick and click on the image above to pick the color. Colors will be displayed in boxes below.")
                    cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
                    if team_1:
                        st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(picked_color)

                    with cp1:
                        hex_color_1 = st.session_state[f"{team1_name} Shirt"] if f"{team1_name} Shirt" in st.session_state else '#FFFFFF'
                        team1_shirt_color = st.color_picker(label=' ', value=hex_color_1, key='t1p', label_visibility='collapsed')
                        st.session_state[f"{team1_name} Shirt"] = team1_shirt_color
                    with cp2:
                        hex_color_2 = st.session_state[f"{team1_name} Short"] if f"{team1_name} Short" in st.session_state else '#FFFFFF'
                        team1_short_color = st.color_picker(label=' ', value=hex_color_2, key='t1gk', label_visibility='collapsed')
                        st.session_state[f"{team1_name} Short"] = team1_short_color
                    with cp3:
                        hex_color_3 = st.session_state[f"{team1_name} Socks"] if f"{team1_name} Socks" in st.session_state else '#FFFFFF'
                        team1_socks_color = st.color_picker(label=' ', value=hex_color_3, key='t2p', label_visibility='collapsed')
                        st.session_state[f"{team1_name} Socks"] = team1_socks_color
                    with cp4:
                        hex_color_4 = st.session_state[f"{team1_name} GK"] if f"{team1_name} GK" in st.session_state else '#FFFFFF'
                        team1_gk_color = st.color_picker(label=' ', value=hex_color_4, key='t2gk', label_visibility='collapsed')
                        st.session_state[f"{team1_name} GK"] = team1_gk_color

                    team_2 = st.checkbox('Team two', key='t2' )
              
                    active_color_ = st.radio(label="Select which team color to pick from the image above", options=radio_options_2, horizontal=True,
                                        help="Chose team color you want to pick and click on the image above to pick the color. Colors will be displayed in boxes below.")
                    

                    cp11, cp12, cp13, cp14 = st.columns([1,1,1,1]) 
                    if team_2:
                        st.session_state[f"{active_color_}"] = '#%02x%02x%02x' % tuple(picked_color)
                    with cp11:
                        hex_color_1 = st.session_state[f"{team2_name} Shirt"] if f"{team2_name} Shirt" in st.session_state else '#FFFFFF'
                        team2_shirt_color = st.color_picker(label=' ', value=hex_color_1, key='t11p', label_visibility='collapsed')
                        st.session_state[f"{team2_name} Shirt"] = team2_shirt_color
                    with cp12:
                        hex_color_2 = st.session_state[f"{team2_name} Short"] if f"{team2_name} Short" in st.session_state else '#FFFFFF'
                        team2_short_color = st.color_picker(label=' ', value=hex_color_2, key='t11gk', label_visibility='collapsed')
                        st.session_state[f"{team2_name} Short"] = team2_short_color
                    with cp13:
                        hex_color_3 = st.session_state[f"{team2_name} Socks"] if f"{team2_name} Socks" in st.session_state else '#FFFFFF'
                        team2_socks_color = st.color_picker(label=' ', value=hex_color_3, key='t211p', label_visibility='collapsed')
                        st.session_state[f"{team2_name} Socks"] = team2_socks_color
                    with cp14:
                        hex_color_4 = st.session_state[f"{team2_name} GK"] if f"{team2_name} GK" in st.session_state else '#FFFFFF'
                        team2_gk = st.color_picker(label=' ', value=hex_color_4, key='t211gk', label_visibility='collapsed')
                        st.session_state[f"{team2_name} GK"] = team2_gk
                
                    colors_dic, color_list_lab = create_colors_info(team1_name, st.session_state[f"{team1_name} Shirt"], st.session_state[f"{team1_name} Short"], st.session_state[f"{team1_name} Socks"], st.session_state[f"{team1_name} GK"],
                                    team2_name, st.session_state[f"{team2_name} Shirt"], st.session_state[f"{team2_name} Short"], st.session_state[f"{team2_name} Socks"], st.session_state[f"{team2_name} GK"],)

                    predict_team_color = st.button('Predict Team', key='team_prediction')
                    save_colors = st.button('Save Colors', key='team_colors')

                    if predict_team_color:
                        team_prediction = predict_team(imgs_list, colors_dic, color_list_lab)
                        team_lables = [f'Team {i}' for i in team_prediction]
                        xy = detection.xyxy
                        class_id = detection.class_id
                        for i in range(len(team_lables)):
                            x = int(xy[i][0])
                            y = int(xy[i][1])
                            label = [ f'Id {object_id}' for object_id in detection.tracker_id]
                            cls_id = class_id[i]
                            if team_lables[i] == 'Team 0':
                                cv2.putText(frame, str(f'{team1_name} - {label[i]}'), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            elif team_lables[i] == 'Team 1':
                                if cls_id == 3:
                                    cv2.putText(frame, str(f'Ref'), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                else:
                                    cv2.putText(frame, str(f'{team2_name} - {label[i]}'), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        ellipse_annotator.annotate( scene=frame, detections=detection)

                        place_holder.image(frame, use_column_width=True, channels="BGR",  caption="Game footage frame",)

                    if save_colors:
                        team_1_color = f'{st.session_state[f"{team1_name} Shirt"]}+{st.session_state[f"{team1_name} Short"]}+{st.session_state[f"{team1_name} Socks"]}+{st.session_state[f"{team1_name} GK"]}'
                        team_2_color = f'{st.session_state[f"{team2_name} Shirt"]}+{st.session_state[f"{team2_name} Short"]}+{st.session_state[f"{team2_name} Socks"]}+{st.session_state[f"{team2_name} GK"]}'
                        
                        st.write(team1_name, team_1_color, team2_name, team_2_color)
                        update_match(int(match_id), team_1_color, team_2_color)


@st.cache_resource(experimental_allow_widgets=True)
def match_info(processed_vid, tempfSecond, video_meta_00, match_df):
    # processed_vid, tempfSecond, video_meta_00 = video_uploader('newww')
    if processed_vid:
        frame_nbr = st.slider(label="Select frame", min_value=1, max_value=video_meta_00[3], step=1, help="Select frame to analyze", key='player_location')

        frame = return_frame(tempfSecond.name, frame_nbr)



        tdcp1, tdcp4 = st.columns([0.55,0.45])

        st.markdown('---')
        preview_frame = st.empty()
        with tdcp1:
            extracted_frame = st.empty()
            extracted_frame.image(frame, use_column_width=True, channels="BGR",  caption="Game footage frame",)


        with tdcp4:
            # 
            st.write("Add Match Information")
            table_form(match_df, processed_vid.name, frame, frame_nbr, preview_frame)
 
            





            
