import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from functions.calibration import calibrate, undistort_frame
from database.db import read_table, save_moment, update_detections, save_corners, Corners, Match, Detection
# from database import db
from functions.model import run_inference, run_inferene_on_frame
from resources.widegts import return_frame, video_uploader
from resources.app_functions import predict_team, create_colors_info, crop_detections, get_pixel_color, grid_image
import supervision as sv

map_ver_path = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/pitch_map/map_ver.png'
map_ver_points_path = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/working_frames/pitch_map/map_ver_with_points.png'
df = pd.read_excel('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/functions/corners_green.xlsx')
model_path = '/Users/abenezer/Downloads/best (16).pt'

ellipse_annotator = sv.EllipseAnnotator()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

@st.cache_resource(experimental_allow_widgets=True)
def tactical_map(processed_vid, tempfSecond, video_meta_00):
    # processed_vid, tempfSecond, video_meta_00 = video_uploader('newjjww')

    if processed_vid:
        cap_temp = cv2.VideoCapture(tempfSecond.name)
        
        points_entered = []
        frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_nbr = st.slider(label="Select frame", min_value=1, max_value=frame_count, step=1, help="Select frame to analyze", key='framcce0')

        cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
        success, frame = cap_temp.read()

        tp1, tp2, tp3, tp4, tp5 = st.columns([1,1,1,1,1])

        with tp1:
            x = st.number_input("X Coordinate", step=1)
        with tp2:
            y = st.number_input("Y Coordinate", step=1)
        with tp3:
            p = st.selectbox("Corresponding Map Point", options=[_ for _ in range(1, 40)])   
        with tp4:
                if st.button("Add Point"):
                    save_corners(processed_vid.name, int(x), int(y), int(p), frame_nbr)
                    # points_entered.append((x, y))
        with tp4:
            plot_points =  st.button("Plot Point", key='plot_btn')
            if plot_points:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, str(p), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        with tp5:
            trasform =  st.button("Apply Transformation", key='trasform')

            preview =  st.button("Download Image", key='preview')
        
        



        tdcp1, tdcp4 = st.columns([0.75,0.25])
        with tdcp1:
            extracted_frame = st.empty()
            extracted_frame.image(frame, use_column_width=True, channels="BGR",  caption="Game footage frame",)
        with tdcp4:
            map = cv2.imread(map_ver_path)
            map_02 = cv2.imread(map_ver_points_path)
            tact_map = st.empty()
            tact_map.image(map_02, use_column_width=True, channels="BGR",  caption="Tactical Map")
            # st.image('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/pitch ver with corners.png', use_column_width=True, channels="BGR",  caption="Tactical Map")


        
        st.markdown('---')

        tsdcp1, tsdcp4 = st.columns([0.75,0.25])
        with tsdcp1:
            preview_windwos = st.empty()
        with tsdcp4:
            tactical = st.empty()
        st.markdown('---')

        tablecp1, tablecp2, tablecp4 = st.columns([1,1,1])

        with tablecp1:
            st.subheader("Corner Points")
            st.data_editor(df)

        with tablecp2:
            st.subheader("Points Entered")
            df_points = read_table(Corners)
            if not df_points.empty:
                df_points = df_points[df_points['vid_name']==processed_vid.name]
            st.write(df_points)

        with tablecp4:
            st.subheader("Matching Points")
            if not df_points.empty:
                match_df = pd.merge(df_points, df, on='point', how='inner')
                st.data_editor(match_df[['point', 'x_x', 'y_x', 'x_y', 'y_y']])
                # st.write(match_df.columns)
                keypoints_1 = match_df[['x_x', 'y_x']].values
                # st.write(keypoints_1)
                keypoints_2 = match_df[['x_y', 'y_y']].values
                # st.write(keypoints_2)

        if trasform:
            if not df_points.empty:
                H, _ = cv2.findHomography(keypoints_1, keypoints_2, cv2.RANSAC)
                warped_img = cv2.warpPerspective(frame, H, (map.shape[1], map.shape[0]))
                preview_windwos.image(warped_img, channels="BGR",)

                df_match = read_table(Match) 
                if not df_match.empty:
                    df_match = df_match[df_match['video']==processed_vid.name]
                    team1_name = df_match['teamA'].values[0]
                    team2_name = df_match['teamB'].values[0]
                    team1colors = df_match['teamA_Colors'].values[0]
                    t1colors = team1colors.split('+')
                    team2colors = df_match['teamB_Colors'].values[0]
                    t2colors = team2colors.split('+')
                    match_id = int(df_match['id'].values[0])
                    # with st.spinner('Detecting players in selected frame..'):
                    df_detections = read_table(Detection)

                    if not df_detections.empty:
                        df_detections = df_detections[(df_detections['video_id'] == int(match_id))&(df_detections['frame'] == 'frame_nbr')]
                    if df_detections.empty:
                        image, detection, imgs_list = run_inferene_on_frame(model_path, frame, processed_vid.name, frame_nbr)
                    else:
                        imgs_list = crop_detections(frame, df_detections)

                    colors_dic, color_list_lab = create_colors_info(team1_name, t1colors[0], t1colors[1], t1colors[2], t1colors[3],
                                                                    team2_name, t2colors[0], t2colors[1], t2colors[2], t2colors[3],)


                    team_prediction = predict_team(imgs_list, colors_dic, color_list_lab)
                    team_lables = [f'Team {i}' for i in team_prediction]
                    xy = detection.xyxy
                    class_id = detection.class_id
                    transformed_people = []
                    for i in range(len(xy)):
                        x = int(xy[i][0])
                        y = int(xy[i][1])
                        x_c = int((xy[i][0] + xy[i][2])/2)
                        y_b = int(xy[i][3])
                        cls_id = class_id[i]
                        label = [ f'Id {object_id}' for object_id in detection.tracker_id]
                        point = np.array([[x_c], [y_b], [1]])

                        transformed_point = np.dot(H, point)
                        transformed_point = (int(transformed_point[0]/transformed_point[2]), int(transformed_point[1]/transformed_point[2]))
                        transformed_people.append(transformed_point)
                        

                        if (cls_id == 3) or (cls_id == 4):
                            cv2.putText(frame, str(f'Ref'), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.circle(map, (transformed_point[0], transformed_point[1]), 10, (255, 255, 0), -1)
                        elif cls_id == 0:
                            cv2.putText(frame, str(f'Ball'), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.circle(map, (transformed_point[0], transformed_point[1]), 10, (0, 0, 0), -1)
                        # elif cls_id == 1:
                        #     cv2.putText(frame, str(f'Ball'), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        #     cv2.circle(map, (transformed_point[0], transformed_point[1]), 10, (0, 0, 0), -1)
                        else:
                            if team_lables[i] == 'Team 0':
                                cv2.putText(frame, str(f'{team1_name} - {label[i]}'), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                cv2.circle(map, (transformed_point[0], transformed_point[1]), 10, (230, 44, 21), -1)
                            elif team_lables[i] == 'Team 1':
                                cv2.putText(frame, str(f'{team2_name} - {label[i]}'), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                                cv2.circle(map, (transformed_point[0], transformed_point[1]), 10, (12, 79, 255), -1)

                    ellipse_annotator.annotate( scene=frame, detections=detection)

                    preview_windwos.image(frame, use_column_width=True, channels="BGR",  caption="Game footage frame",)
                    tactical.image(map, channels="BGR",)
                    # st.write(colors_dic, color_list_lab)




                # # image, detection = run_inferene_on_frame(model_path, frame)
                # preview_windwos.image(image, channels="BGR",)

                # bboxes = detection.xyxy
                # xy = []

                # for i in range(len(bboxes)):
                #     x = (bboxes[i][0] + bboxes[i][2])/2
                #     y = bboxes[i][3]
                #     xy.append((round(x), round(y)))
                # # Transform detected people coordinates using homography
                # transformed_people = []
                # for (x, y) in xy:
                #     # Convert pixel coordinates to homogeneous coordinates
                #     point = np.array([[x], [y], [1]])
                #     # Apply homography transformation
                #     transformed_point = np.dot(H, point)
                #     # Convert back to Cartesian coordinates
                #     transformed_point = (int(transformed_point[0]/transformed_point[2]), int(transformed_point[1]/transformed_point[2]))
                #     transformed_people.append(transformed_point)
                #     cv2.circle(map, (transformed_point[0], transformed_point[1]), 10, (0, 0, 255), -1)
                # tactical.image(map, channels="BGR",)
                # st.write(H)
        cap_temp.release()
        # if preview:
            # warped_img = cv2.warpPerspective(frame, H, (map.shape[1], map.shape[0]))
            # preview_windwo.image(warped_img, channels="BGR",)
            