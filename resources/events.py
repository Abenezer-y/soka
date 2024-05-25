from database.db import read_table, save_moment, update_detections, Event, Match, Detection, shot, attacking_actions, set_pieces, pass_events, defensive_events, aerial_actions
import streamlit as st
import cv2
import numpy as np
from resources.tab_2 import game_page
from resources.widegts import return_frame, video_uploader
from functions.video_preprocess import video_meta, trim_video
from functions.model import run_inference, run_inferene_on_frame
from functions.filters import equqlize_hist
from resources.app_functions import play_video, load_moment
import os
import pandas as pd

event_type = ['Goal', 'Yellow Card', 'Red Card', 'Substitute', 'Foul', 'Shots', 'Set Pieces', 'Pass', 'Defensive Action', 'Aerial', 'Attacking Actions']
HOME_PATH = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/moments/'
model_path = '/Users/abenezer/Downloads/best (16).pt'

def add_players_info(df, player_options):

    with st.expander("Team Detection"):
        ex_cols = st.columns(9)

        with ex_cols[0]:
            st.write('Select Player ID')

        with ex_cols[1]:
            player_id = st.selectbox('-', player_options, label_visibility='collapsed', key='player_id_select')

        with ex_cols[2]:
            st.write('Club')

        with ex_cols[3]:
            club_name = st.selectbox('Teams', ['clubs'], label_visibility='collapsed', key='tclub')

        with ex_cols[8]:
            save_player = st.button('Update Team', key='btn_update_detection')

    if save_player:
        data = df[df['object_id']==player_id]
        ids = data['id'].values.tolist()
        st.write(ids)
        # for id in ids:
        #     update_detections(id, player_name, club_name, jersey)

def edit_players_id(df, player_options, frm_nm):
    st.session_state["starting_frame"] = st.session_state["starting_frame"]  if  "starting_frame" in st.session_state else 1
    st.session_state["ending_frame"] = st.session_state["ending_frame"]  if "ending_frame" in st.session_state else 1

    with st.expander("Player Detection"):
        ex_cols = st.columns(9)

        with ex_cols[0]:
            st.write('Select Player ID')

        with ex_cols[1]:
            player_id = st.selectbox('-', player_options, label_visibility='collapsed', key='player_id_select_for')

        with ex_cols[2]:
            st.write('Club: ')

        with ex_cols[3]:
            st.write('Club Name')
            # club_name = st.selectbox('Teams', ['clubs'], label_visibility='collapsed', key='tclub')
        with ex_cols[4]:
            starting_frame = st.checkbox('Starting frame', key='Starting')
            if starting_frame: 
                frm = frm_nm
                st.session_state["starting_frame"] = frm_nm
            start_f = st.number_input('-', 0, 1000, st.session_state["starting_frame"], disabled=~starting_frame, label_visibility='collapsed', key='start_frame')
                

            # st.write(st.session_state["starting_frame"])

        with ex_cols[5]:
            ending_frame = st.checkbox('Ending frame', key='Ending')
            if ending_frame: 
                st.session_state["ending_frame"] = frm_nm
            end_f = st.number_input('-', 0, 1000, st.session_state["ending_frame"], disabled=~ending_frame, label_visibility='collapsed', key='end_frame')
                
            # st.write(st.session_state["ending_frame"])

        with ex_cols[6]:
            st.write('New ID')

        with ex_cols[7]:
            new_id = st.number_input(label='-', label_visibility='collapsed', value=0, key='new_id')

        with ex_cols[8]:
            save_player = st.button('Update ID', key='btn_update_id_detection')

    if save_player:
        data = df[df['object_id']==player_id]
        ids = data['id'].values.tolist()
        st.write(ids, new_id)
        # for id in ids:
        #     update_detections(id, player_name, club_name, jersnew_idey)

def add_moment_info(match_id, moment_id, fr_nm):
    with st.expander("Event Deatils"):

        r1 = st.columns(9)

        with r1[0]:
            event_type_select = st.selectbox('-', event_type, label_visibility='collapsed', key='event_type_key', help="Select Action")
        
        with r1[1]:
            club_name = st.text_input(label='-', label_visibility='collapsed', key='c_nam', placeholder="Add Club")

        with r1[2]:
            player_name = st.text_input(label='-', label_visibility='collapsed', key='p_na', placeholder="Add Player")

        with r1[4]:
            if event_type_select == 'Shots':
                opts = st.selectbox('-', shot, label_visibility='collapsed', key='shots_option')
            elif event_type_select == 'Set Pieces':
                opts = st.selectbox('-', set_pieces, label_visibility='collapsed', key='set_peices')
            elif event_type_select == 'Pass':
                opts = st.selectbox('-', pass_events, label_visibility='collapsed', key='set_peices')
            elif event_type_select == 'Defensive Action':
                opts = st.selectbox('-', defensive_events, label_visibility='collapsed', key='set_peices')
            elif event_type_select == 'Attacking Actions':
                opts = st.selectbox('-', attacking_actions, label_visibility='collapsed', key='set_peices')
            elif event_type_select == 'Aerial':
                opts = st.selectbox('-', aerial_actions, label_visibility='collapsed', key='set_peices')
            else:
                opts = 'N/A'

        with r1[8]:
            save_player = st.button('Save', key='save_match_details')

        if save_player:
            st.write(club_name, player_name, fr_nm, opts, match_id, moment_id)

def game_page(processed_vid, video,  video_meta_00, s_time, e_time, id):

    if processed_vid:

        row_two = st.columns(8)
        with row_two[0]:
            is_playing = st.checkbox("Play (@ 0.15 speed)", value=False, key='play')   
        with row_two[1]:
            is_annotate = st.checkbox("Annotate", value=False, key='annto')  

        row_one = st.columns([1, 0.1])
        cols_an = st.columns([0.7, 0.3])

        with cols_an[0]:
            image_holder = st.empty()
            
        if is_playing:
            play_video(video, image_holder, is_playing)
        else:
            with row_one[1]:
                frame_input = st.number_input('_', s_time, e_time, s_time, label_visibility='collapsed')
            with row_one[0]:
                frame_nbr = st.slider(label="Select frame", value=frame_input, min_value=s_time, max_value=e_time, step=1, help="Select frame to analyze", key='frame_number')
                frame = return_frame(video, frame_nbr)
                image_holder.image(frame, channels='BGR')




        data_detections = read_table(Detection)
        if not data_detections.empty:
            df = data_detections[data_detections['video_id']==id]

            player_id_options = df[df['frame']==frame_nbr]['object_id'].values.tolist()


            add_players_info(df, player_id_options)
            edit_players_id(df, player_id_options, frame_nbr)
            add_moment_info(1, 1, frame_nbr)


            buttn_dis = False
        else:
            df = []
            buttn_dis = False

        with cols_an[1]:
            st.data_editor(df)

        with row_two[7]:
            run_detection = st.button("Run Detection", disabled=buttn_dis, key='detection_model') 
            if run_detection:
                out_file = '/Users/abenezer/Documents/Game Footage/Ethiopia Bunna/projcet/vidoe.mp4'
                frame = equqlize_hist(frame)
                image, detection, imgs_list = run_inferene_on_frame(model_path, frame, processed_vid.name, frame_nbr)
                image_holder.image(image, channels='BGR')
                # run_inference(video, id, out_file, model_path, start=s_time, end=e_time)

def event_page( name, video, processed_vid, tempfSecond, video_meta_00, project_folder):
        tab1, tab2 = st.tabs(['Event Identifying', 'Event Details'])
        df = read_table(Match)
        data_df = read_table(Event)



        if not df.empty:
            df = df[df['video']==name]
            if not df.empty:
                team1 = df['teamA'].values[0]
                team2 = df['teamB'].values[0]
                id = df['id'].values[0]
                st.write(id)
                if not data_df.empty:
                    data = data_df[data_df['match_id']==id]
                else:
                    data = pd.DataFrame([])
            else:
                team1 = 'Team 1'
                team2 = 'Team 2'
      
                data = pd.DataFrame([])
        else:
            team1 = 'Team 1'
            team2 = 'Team 2'
         
            data = pd.DataFrame([])

        with tab1:    
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.title("Event Identifiying")
                if video:
                    video_player = st.video(video, start_time=0)
        
                col_112, col_111, col_11, col_12, col_13, col_14, col_15, col_16, col_17, col_18 = st.columns( [0.16, 0.16, 0.14, 0.06, 0.09, 0.1, 0.05, 0.07, 0.07, 0.1])
                with col_112:
                    club = st.selectbox('1', options=[team1, team2], label_visibility='collapsed', key='team')
                with col_111:
                    events = st.selectbox('1', options=['Build Up', 'Defense', 'Transition A to D', 'Transition D to A', 'Set Pieces'], label_visibility='collapsed', key='moments')
                with col_11:
                    half = st.selectbox('1', options=['1st Half', '2nd Half'], label_visibility='collapsed', key='half')
                with col_12:
                    st.write('Start')
                with col_13:
                    min_start = st.number_input('1', min_value=0, max_value=90, value=0, step=1, key='Start_min', label_visibility='collapsed', )
                with col_14:
                    sec_start = st.number_input('1', min_value=0, max_value=59, value=0, step=1, key='Start_sec', label_visibility='collapsed', )
                with col_15:
                    st.write('End')
                with col_16:
                    min_end = st.number_input('1', min_value=0, max_value=90, value=0, step=1, key='end_min', label_visibility='collapsed')
                with col_17:
                    sec_end = st.number_input('1', min_value=0, max_value=59, value=0, step=1, key='end_sec', label_visibility='collapsed')
                with col_18:
                    save_event = st.button('Save')
                    if save_event:
                        # st.write(id)
                        save_moment(club ,half, events, min_start, sec_start, min_end, sec_end, int(id))
                        # update_video(video_id, team_1, team_2)
                        data_df = read_table(Event)
                        data = data_df[data_df['match_id']==id]

            with col2:
                st.title('Moments')
                st.data_editor(data)

            st.markdown('---')
        with tab2:  
            row_two = st.columns([0.2,0.2,1,0.2,0.2])
            if not data.empty:
                moments = data['name'].unique().tolist()
                ids = data['id'].unique().tolist() 
                with row_two[0]:
                    moment = st.selectbox(' ', options=moments, key='moments_secet', label_visibility='collapsed')
                with row_two[1]:
                    moment_ids = st.selectbox(' ', options=ids, key='moments_ids', label_visibility='collapsed')

                with row_two[3]:
                    load_event = st.checkbox('Load Moment', key='load_event')

                with row_two[4]:
                    play = st.checkbox('Play Moment', key='play_moment')


                if load_event:
                    df_mom = data[data['id']==int(moment_ids)]
                    sm = df_mom['start_min'].values[0]
                    ss = df_mom['start_sec'].values[0]
                    em = df_mom['end_min'].values[0]
                    es = df_mom['end_sec'].values[0]    
                                # convert_to_frame_count()       
                    s_time = (60*(sm) + ss)*video_meta_00[0]
                    e_time = (60*(em) + es)*video_meta_00[0]
                    file = os.path.join(project_folder, f'{moment}_{moment_ids}.mp4')


                    load_moment(tempfSecond.name, file, sm, ss, em, es)
                    if play:
                        st.video(file)
                    else:
                        st.write(s_time, e_time)
                        game_page(processed_vid, file, video_meta_00, s_time, e_time, id)