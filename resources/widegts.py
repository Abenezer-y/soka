from functions.video_preprocess import video_meta
import streamlit as st
import cv2
import tempfile
from database.db import read_match

## Sidebar Setup
def video_uploader(key='new'):
    st.sidebar.markdown('---')
    st.sidebar.subheader("Video Upload")
    input_vide_file = st.sidebar.file_uploader('Upload a video file', type=['mp4','mov', 'avi', 'm4v', 'asf'], key=key)
    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=True)
    
    if input_vide_file:
        tempf.write(input_vide_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()
        fps, width, height, frames_count = video_meta(tempf.name)
        video_info = (fps, width, height, frames_count)
        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)
        st.sidebar.markdown('---')
        df = read_match(input_vide_file.name)
        return input_vide_file, tempf, video_info, df 
    
    return None, None, None, None

    


def return_frame(file, frame_nbr):
    cap_temp = cv2.VideoCapture(file)
    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
    success, frame = cap_temp.read()
    cap_temp.release()
    return frame
