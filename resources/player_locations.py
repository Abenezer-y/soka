import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from functions.calibration import calibrate, undistort_frame
from database.db import read_table, save_moment, update_detections, save_corners, Corners
# from database import db
from functions.model import run_inference, run_inferene_on_frame
from streamlit_image_coordinates import streamlit_image_coordinates
from inference_sdk import InferenceHTTPClient
import supervision as sv


# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2U7K8nQzMwcUMMgbI9Yj"
)

model_path = '/Users/abenezer/Downloads/best (16).pt'

@st.cache_resource(experimental_allow_widgets=True)
def detection(tempfSecond,name,  team1_name='Team A', team2_name='Team B'):
    demo_team_info = {
        "Demo 1":{"team1_name":"France",
                  "team2_name":"Switzerland",
                  "team1_p_color":'#1E2530',
                  "team1_gk_color":'#F5FD15',
                  "team2_p_color":'#FBFCFA',
                  "team2_gk_color":'#B1FCC4',
                  },
        "Demo 2":{"team1_name":"Chelsea",
                  "team2_name":"Manchester City",
                  "team1_p_color":'#29478A',
                  "team1_gk_color":'#DC6258',
                  "team2_p_color":'#90C8FF',
                  "team2_gk_color":'#BCC703',
                  }
    }
   
    selected_team_info = demo_team_info['Demo 1']
   
    cap_temp = cv2.VideoCapture(tempfSecond.name)
    frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_nbr = st.slider(label="Select frame", min_value=1, max_value=frame_count, step=1, help="Select frame to analyze", key='player_location')
    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
    success, frame = cap_temp.read()

    tp1, tp2, tp3, tp4, tp5 = st.columns([1,1,1,1,1])

    # with tp1:
    #     x = st.number_input("X Coordinate", step=1, key)
    # with tp2:
    #     y = st.number_input("Y Coordinate", step=1)
    # with tp3:
    #     p = st.selectbox("Corresponding Map Point", options=[_ for _ in range(1, 40)])   
    # with tp4:
    #         if st.button("Add Point"):
    #             save_corners(name, int(x), int(y), int(p))
    #             # points_entered.append((x, y))

    # with tp4:
    #     plot_points =  st.button("Plot Point", key='button_p')
    #     if plot_points:
    #             cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    #             cv2.putText(frame, str(p), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    with tp5:
        trasform =  st.button("Run Detection", key='detect')

        # preview =  st.button("Plot on tcatical map", key='plot_on')
    
    



    tdcp1, tdcp4 = st.columns([0.75,0.25])
    with tdcp1:
        extracted_frame = st.empty()
        extracted_frame.image(frame, use_column_width=True, channels="BGR",  caption="Game footage frame",)
    # with tdcp4:
    #     map = cv2.imread('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/pitch ver.png')
    #     tactical_map = st.empy()
    #     # # Plot points
    #     # points = df[['x', 'y']].values
    #     # for i, (x, y) in enumerate(points):
    #     #     cv2.circle(map, (x, y), 5, (0, 0, 255), -1)
    #     #     cv2.putText(map, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #     # cv2.imwrite('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/pitch ver with corners.png', map)
    #     st.image('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/pitch ver with corners.png', use_column_width=True, channels="BGR",  caption="Tactical Map")
    #     # tact_map.image(map, use_column_width=True, channels="BGR",  caption="Processed Image",)
    
    st.markdown('---')
    preview_windwo = st.empty()
    st.markdown('---')



    if trasform:
        # image, detection = run_inferene_on_frame(model_path, frame)
        model_id="action-players/16"
        # run inference
        results = CLIENT.infer(frame, model_id=model_id)
        detections = sv.Detections.from_inference(results)
        ellipse_annotator = sv.EllipseAnnotator()
        image = ellipse_annotator.annotate(
            scene=frame.copy(),
            detections=detections
)

        preview_windwo.image(image, channels="BGR",)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections_imgs_list = []
        detections_imgs_grid = []
        padding_img = np.ones((80,60,3),dtype=np.uint8)*255
        bboxes = detection.xyxy
        st.write(len(bboxes))
        for i in range(len(bboxes)):
            bbox = bboxes[i]                         
            obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            obj_img = cv2.resize(obj_img, (60,80))
            detections_imgs_list.append(obj_img)
        detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2)])
        detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2, len(detections_imgs_list))])
        # detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2,(3*len(detections_imgs_list)//4))])
        # detections_imgs_grid.append([detections_imgs_list[i] for i in range((3*len(detections_imgs_list)//4), len(detections_imgs_list))])
        if len(detections_imgs_list)%2 != 0:
            detections_imgs_grid[0].append(padding_img)
        concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
        concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
        # concat_det_imgs_row3 = cv2.hconcat(detections_imgs_grid[2])
        # concat_det_imgs_row4 = cv2.hconcat(detections_imgs_grid[3])
        concat_det_imgs = cv2.vconcat([concat_det_imgs_row1,concat_det_imgs_row2])
        st.write(detection.xyxy)
        with tdcp4:
            st.write("Detected players")
            value = streamlit_image_coordinates(concat_det_imgs, key="numpy")
#value_radio_dic = defaultdict(lambda: None)
        st.markdown('---')
        radio_options =[f"{team1_name} P color", f"{team1_name} GK color",f"{team2_name} P color", f"{team2_name} GK color"]
        active_color = st.radio(label="Select which team color to pick from the image above", options=radio_options, horizontal=True,
                                help="Chose team color you want to pick and click on the image above to pick the color. Colors will be displayed in boxes below.")
        if value is not None:
            picked_color = frame[value['y'], value['x'], :]
            st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(picked_color)
            st.write("Boxes below can be used to manually adjust selected colors.")
            cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
            with cp1:
                hex_color_1 = st.session_state[f"{team1_name} P color"] if f"{team1_name} P color" in st.session_state else selected_team_info["team1_p_color"]
                team1_p_color = st.color_picker(label=' ', value=hex_color_1, key='t1p')
                st.session_state[f"{team1_name} P color"] = team1_p_color
            with cp2:
                hex_color_2 = st.session_state[f"{team1_name} GK color"] if f"{team1_name} GK color" in st.session_state else selected_team_info["team1_gk_color"]
                team1_gk_color = st.color_picker(label=' ', value=hex_color_2, key='t1gk')
                st.session_state[f"{team1_name} GK color"] = team1_gk_color
            with cp3:
                hex_color_3 = st.session_state[f"{team2_name} P color"] if f"{team2_name} P color" in st.session_state else selected_team_info["team2_p_color"]
                team2_p_color = st.color_picker(label=' ', value=hex_color_3, key='t2p')
                st.session_state[f"{team2_name} P color"] = team2_p_color
            with cp4:
                hex_color_4 = st.session_state[f"{team2_name} GK color"] if f"{team2_name} GK color" in st.session_state else selected_team_info["team2_gk_color"]
                team2_gk_color = st.color_picker(label=' ', value=hex_color_4, key='t2gk')
                st.session_state[f"{team2_name} GK color"] = team2_gk_color
            st.markdown('---')