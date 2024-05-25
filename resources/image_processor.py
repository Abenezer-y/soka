from database.db import read_table, save_moment, save_video, update_detections, update_video
import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from functions.video_preprocess import video_meta, trim_video
from database import db
from functions.model import run_inference
from functions.filters import *
from functions.annotation.annotator import annotate_frame
from streamlit_image_coordinates import streamlit_image_coordinates

MODEL_PATH = '/Users/abenezer/Downloads/best (15).pt'
HOME_PATH = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/moments/'
MODEL_PATH = '/Users/abenezer/Downloads/best (15).pt'
HOME_PATH = '/Users/abenezer/Documents/Projects/SOKA/video_analyzer/moments/'

# import streamlit as st
# from os import listdir, path
# import cv2



# Print x y coordinates for left mouse clicks with cv2
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)




def region_of_interest(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Perform histogram equalization only on the V channel, for value intensity.
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # Set range for green color.
    g_lb = np.array( [31, 50, 50] , np.uint8)
    g_ub = np.array( [80, 255, 255], np.uint8)

    g_mask = cv2.inRange(img_hsv, g_lb, g_ub)
    
    masked_image = cv2.bitwise_and(img, img, mask = g_mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Utility for defining Line Segments."""
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]),
        minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines


def separate_left_right_lines(lines):
    """Separate left and right lines depending on the slope."""
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2: # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif y1 < y2: # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines


def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1
        return sum(values) / n


def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines keeping in mind the lower and upper border intersections."""
    slopes = []
    consts = []
    
    if (lines is not None) and (len(lines) != 0):
        for x1, y1, x2, y2 in lines:
            slope = (y1-y2) / (x1-x2)
            slopes.append(slope)
            c = y1 - slope * x1
            consts.append(c)
        avg_slope = cal_avg(slopes)
        avg_consts = cal_avg(consts)

        # Calculate average intersection at lower_border.
        x_lane_lower_point = int((lower_border - avg_consts) / avg_slope)

        # Calculate average intersection at upper_border.
        x_lane_upper_point = int((upper_border - avg_consts) / avg_slope)

        return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]

def process_image(image):  
    # Convert to grayscale.
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Intensity selection.
    # gray_select = cv2.inRange(gray, 150, 255)
    
    # Region masking: Select vertices according to the input image.
    # roi_vertices = np.array([[[100, 540], [900, 540], [525, 330], [440, 330]]])
    select_roi = region_of_interest(image)
    
    # diameter of the pixel neighborhood used during filtering.
    dia = 15

    # Larger the value the distant colours will be mixed together
    # to produce areas of semi equal colors.
    sigmaColor = 150

    # Larger the value more the influence of the farther placed pixels 
    # as long as their colors are close enough.
    sigmaSpace = 80

    blurred_image = cv2.bilateralFilter(select_roi, dia, sigmaColor, sigmaSpace)

    thresh = thresh_img(blurred_image, t_val=51, t_type='Adaptive')
    # Canny Edge Detection.
    low_threshold = 50
    high_threshold = 100
    img_canny = cv2.Canny(blurred_image, low_threshold, high_threshold)
    
    # Remove noise using Gaussian blur.
    # kernel_size = 5
    # canny_blur = cv2.GaussianBlur(img_canny, (kernel_size, kernel_size), 0)
    




    # Hough transform parameters set according to the input image.
    rho = 10
    theta = np.pi/180
    threshold = 100
    min_line_len = 50
    max_line_gap = 300

    hough, lines = hough_lines(img_canny, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Extrapolate lanes.
    roi_upper_border = 330
    roi_lower_border = 540
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)
    lane_img = extrapolated_lane_image(image, lines, roi_upper_border, roi_lower_border)
    
    # Combined using weighted image.
    image_result = cv2.addWeighted(image, 1, lane_img, 0.4, 0.0)
    return image_result


st.cache_resource
def game_page(video, video_id):
    data_moments = read_table(db.Event)

    if  not data_moments.empty:
        df = data_moments[data_moments['video_id']==video_id]
        event_options = df['name'].unique().tolist()
    


        analyzer_cols = st.columns(8)
        filter_options = [None, 'Find Feild Borders']
        with analyzer_cols[0]:
            event = st.selectbox('-', options=event_options, label_visibility='collapsed', key='event_test')
            
        with analyzer_cols[1]:
            df_filter = df[df['name'] == event]
            indecies = df_filter.index.tolist()
            moments_option = [f'Moment {_ + 1}' for _ in range(df_filter.shape[0])]

            momet = st.selectbox('-', options=moments_option, label_visibility='collapsed', key='momet_count_test')
            index = moments_option.index(momet)
            i = indecies[index]
            momet_id = df_filter['id'][i]
        input_file = HOME_PATH +f'{video.name}' + f'/{event}_{momet}.mp4'
        video_capture = cv2.VideoCapture(input_file)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        with analyzer_cols[2]:
            is_playing = st.checkbox("Play (@ 0.25 speed)", value=False, key='play_test')   
        with analyzer_cols[3]:
            filter = st.selectbox("_", filter_options, key='filters', label_visibility='collapsed') 

        with analyzer_cols[4]:
            is_annotate = st.checkbox("Annotate", value=False, key='annto_test')   

        with analyzer_cols[5]:
            st.write('Frames: ')

        with analyzer_cols[6]:
            frame_input = st.number_input('_', 1, total_frames, 1, label_visibility='collapsed')


            
        frame_slider = st.slider('level', 1, total_frames, frame_input, label_visibility='collapsed')

        data_detections = read_table(db.Detection)
        data_video = read_table(db.Video)
        df = data_detections[(data_detections['event_id']==momet_id) & (data_detections['video_id']==video_id) & (data_detections['frame']==frame_slider)]
        player_id_options = df['object_id'].values.tolist()

        df_vid = data_video[data_video['id']==video_id]
        clubs = [None, df_vid['club_srl'].unique()[0], df_vid['club_slr'].unique()[0]]

        with st.expander("Add Player and Club Names"):
            ex_cols = st.columns(9)

            with ex_cols[0]:
                st.write('Select Player ID')

            with ex_cols[1]:
                player_id = st.selectbox('-', player_id_options, label_visibility='collapsed', key='player_id_select')

            with ex_cols[2]:
                st.write('Club')

            with ex_cols[3]:
                club_name = st.selectbox('-', clubs, label_visibility='collapsed', key='tclub')

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
        cols_row_2 = st.columns([0.7, 0.3])

        with cols_an[0]:
            image_holder = st.empty()
        with cols_an[1]:
            st.data_editor(df)



        while video_capture.isOpened():

            if is_playing:
                ret, frame = video_capture.read()
                cv2.setMouseCallback('frame', click_event)
                if not ret:
                    break
                
                if filter == 'Find Feild Borders':
                    frame = find_field_lines(frame, gap=250, minLen=250)
                if is_annotate:
                    df = data_detections[(data_detections['event_id']==momet_id) & (data_detections['video_id']==video_id) & (data_detections['frame']==frame_slider)]
                    frame = annotate_frame(frame, df)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_holder.image(frame, channels="RGB")
                frame_slider += 1
                if frame_slider >= total_frames:
                    frame_slider = 0
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
            
            else:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
                ret, frame = video_capture.read()
                cv2.setMouseCallback('frame', click_event)
                if filter == 'Find Feild Borders':
                    frame = find_field_lines(frame, gap=250, minLen=350)
                if is_annotate:
                    frame = annotate_frame(frame, df)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not ret:
                    break
                cv2.setMouseCallback('frame', click_event)
                
                with cols_row_2[0]:
                    value = streamlit_image_coordinates(frame, width=1080, key="numpy",)
                    st.write(value)
                with cols_row_2[1]:
                    value = streamlit_image_coordinates('/Users/abenezer/Documents/Projects/SOKA/video_analyzer/pitch ver.png', width=390,)
                    st.write(value)
                # image_holder.image(frame, channels="RGB")
                st.stop()

        video_capture.close()



def filter_response(filter, image):

    if filter == 'Find Feild Borders':
        frame = find_field_lines(image, gap=250, minLen=250)

    elif filter == 'Adaptive Thresholding':
        frame = find_field_lines(image, gap=250, minLen=250)
    
    return frame
    