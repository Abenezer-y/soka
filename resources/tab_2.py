from database.db import read_table, save_moment, update_detections
import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from functions.video_preprocess import video_meta, trim_video
from database import db
from functions.model import run_inference
from functions.filters import *
from annotation.annotator import annotate_frame
from resources.widegts import return_frame, video_uploader


MODEL_PATH = '/Users/abenezer/Downloads/best (15).pt'
HOME_PATH = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/moments/'

st.cache_resource
def video_player_load(video_capture, video_path, s_min, s_sec, e_min, e_sec):
        page_container = st.container()

        with page_container:

            if not os.path.exists(video_path): 
                trim_video(video_capture, s_min, s_sec, e_min, e_sec, video_path)
                st.video(video_path)
            else:
                st.video(video_path)

st.cache_resource
def video_processor(video_id, folder_path, video):

    data_moments = read_table(db.Event)
    
    event_options = []
    if  not data_moments.empty:
        df = data_moments[data_moments['video_id']==video_id]
        event_options = df['name'].unique().tolist()
    
        col6, col7, col8, col9, col10, col11, col12, col13, col14 = st.columns([0.19, 0.15, 0.1, 0.05, 0.1, 0.05, 0.14,0.12, 0.10])

        with col6:
            event = st.selectbox('-', options=event_options, label_visibility='collapsed', key='event')
        
        with col7:
            df_filter = df[df['name'] == event]
            indecies = df_filter.index.tolist()
            moments_option = [f'Moment {_ + 1}' for _ in range(df_filter.shape[0])]

            momet = st.selectbox('-', options=moments_option, label_visibility='collapsed', key='momet_count')
            index = moments_option.index(momet)
            i = indecies[index]
        with col8:
            st.write('Start time')
            
        with col9:
            s_min = df_filter['start_min'][i]
            s_sec = df_filter['start_sec'][i]
            st.write(f'{s_min}:{s_sec}')

        with col10:
            # end = st.text_input("Enter tracking id to be changed: ", label_visibility='collapsed', key='end')
            e_min = df_filter['end_min'][i]
            e_sec = df_filter['end_sec'][i]
            st.write('End time')
        with col11:
            st.write(f'{e_min}:{e_sec}')
            # save_btn = st.button("Save Event", key='btn_save')
        with col12:
            disable = True
            value_play = False
            load = st.button("Load Moment", key='btn_load')

        with col13:
            run_analysis = st.button("Run Detection", key='btn_detection')
        #     is_playing = st.checkbox("Play", value=value_play, key='play')
        with col14:
            analysis = st.checkbox("Analayze", value=False,key='btn_analyze')

        if load:
            filename = f'{event}_{momet}.mp4'
            path = os.path.join(folder_path, filename)
            video_player_load(video, path, s_min,s_sec, e_min, e_sec)

        if run_analysis:
            input_file = HOME_PATH +f'{video.name}' + f'/{event}_{momet}.mp4'
            file = HOME_PATH +f'{video.name}' + f'/annotated_{event}_{momet}.mp4'
            df_filter = df[df['name'] == event]
            index = moments_option.index(momet)
            i = indecies[index]
            momet_id = df_filter['id'][i]

            ra_col1, ra_col2 = st.columns([0.7,0.3])

            if os.path.exists(file): 
                data_detections = read_table(db.Detection)
                with ra_col1:
                    st.video(data=file)
                with ra_col2:
                    df = data_detections[(data_detections['event_id']==momet_id) & (data_detections['video_id']==video_id)]
                    st.data_editor(data_detections)
            else:
                run_inference(input_video=input_file, video_id=video_id, moment_id=momet_id, out_dir=file, model_path=MODEL_PATH)
                with ra_col1:
                    st.video(data=file)
                with ra_col2:
                    data_detections = read_table(db.Detection)

                    df = data_detections[(data_detections['event_id']==momet_id) & (data_detections['video_id']==video_id)]
                    st.data_editor(df)

        if analysis:
            df_filter = df[df['name'] == event]
            index = moments_option.index(momet)
            i = indecies[index]
            momet_id = df_filter['id'][i]

            game_page(event, video, momet, momet_id, video_id)

st.cache_resource
def game_page(processed_vid, video,  video_meta_00):

    if processed_vid:
        frame_nbr = st.slider(label="Select frame", min_value=1, max_value=video_meta_00[3], step=1, help="Select frame to analyze", key='frame_number')

        frame = return_frame(video.name, frame_nbr)

        analyzer_cols = st.columns(8)
        filter_options = [None, 'Find Feild Borders']

        with analyzer_cols[0]:
            is_playing = st.checkbox("Play (@ 0.15 speed)", value=False, key='play')   
        with analyzer_cols[1]:
            filter = st.selectbox("_", filter_options, key='filters', label_visibility='collapsed') 
        with analyzer_cols[3]:
            is_annotate = st.checkbox("Annotate", value=False, key='annto')   

        with analyzer_cols[4]:
            st.write('Frames: ')

        with analyzer_cols[5]:
            frame_input = st.number_input('_', 1, frame_nbr, 1, label_visibility='collapsed')

        data_detections = read_table(db.Detection)
        # data_video = read_table(db.Video)
        if not data_detections.empty:
            # df = data_detections[(data_detections['event_id']==momet_id) & (data_detections['video_id']==video_id) & (data_detections['frame']==frame_nbr)]
            df = data_detections
            player_id_options = df['object_id'].values.tolist()
        else:
            df = []

        # df_vid = data_video[data_video['id']==video_id]
        # clubs = [None, df_vid['club_srl'].unique()[0], df_vid['club_slr'].unique()[0]]

        with st.expander("Add Player and Club Names"):
            ex_cols = st.columns(9)

            with ex_cols[0]:
                st.write('Select Player ID')

            with ex_cols[1]:
                player_id = st.selectbox('-', player_id_options, label_visibility='collapsed', key='player_id_select')

            with ex_cols[2]:
                st.write('Club')

            with ex_cols[3]:
                club_name = st.selectbox('-', ['clubs'], label_visibility='collapsed', key='tclub')

            with ex_cols[4]:
                st.write('Name')

            with ex_cols[5]:
                player_name = st.text_input(label='-', label_visibility='collapsed', key='tname')

            with ex_cols[6]:
                st.write('Jersey')

            with ex_cols[7]:
                jersey = st.number_input(label='-', label_visibility='collapsed', value=0, key='tjersey')

            with ex_cols[8]:
                save_player = st.button('Save', key='btn_update_detection')

        if save_player:
            data = df[df['object_id']==player_id]
            ids = data['id'].values.tolist()

            for id in ids:
                update_detections(id, player_name, club_name, jersey)
            

        cols_an = st.columns([0.7, 0.3])

        with cols_an[0]:
            image_holder = st.empty()
            image_holder.image(frame, channels='BGR')
        with cols_an[1]:
            st.data_editor(df)



        # while video_capture.isOpened():

        #     if is_playing:
        #         ret, frame = video_capture.read()
        #         if not ret:
        #             break
                
        #         if filter == 'Find Feild Borders':
        #             frame = find_field_lines(frame, gap=250, minLen=250)
        #         if is_annotate:
        #             df = data_detections[(data_detections['event_id']==momet_id) & (data_detections['video_id']==video_id) & (data_detections['frame']==frame_slider)]
        #             frame = annotate_frame(frame, df)
        #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         else:
        #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         image_holder.image(frame, channels="RGB")
        #         frame_slider += 1
        #         if frame_slider >= total_frames:
        #             frame_slider = 0
        #         video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
            
        #     else:
        #         video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
        #         ret, frame = video_capture.read()
        #         if filter == 'Find Feild Borders':
        #             frame = find_field_lines(frame, gap=250, minLen=350)
        #         if is_annotate:
        #             frame = annotate_frame(frame, df)
        #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         else:
        #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         if not ret:
        #             break
        #         image_holder.image(frame, channels="RGB")
        #         st.stop()

        # video_capture.close()
