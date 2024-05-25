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
from functions.video_preprocess import video_meta, trim_video
from functions.video_preprocess import video_meta
import streamlit as st
import cv2
import tempfile


## Sidebar Setup
@st.cache_resource(experimental_allow_widgets=True)
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
        return input_vide_file, tempf, video_info 
    return None, None, None


def return_frame(file, frame_nbr):
    cap_temp = cv2.VideoCapture(file)
    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
    success, frame = cap_temp.read()
    cap_temp.release()
    return frame

@st.cache_resource(experimental_allow_widgets=True)
def load_moment(input, output, s_min, s_sec, e_min, e_sec):
        if not os.path.exists(output): 
            trim_video(input, s_min, s_sec, e_min, e_sec, output)
        
# @st.cache_resource(experimental_allow_widgets=True)
def play_video(video_name, _image_holder, is_playing):
    video = cv2.VideoCapture(video_name)
    while video.isOpened():
        if is_playing:
            ret, frame = video.read()
            if not ret:
                break
            _image_holder.image(frame, channels="BGR")
    video.release()

@st.cache_resource
def predict_team( imgs_list, colors_dic, color_list_lab):

        nbr_team_colors = len(list(colors_dic.values())[0])                                     # Convert frame to RGB
        obj_palette_list = []                                                                   # Initialize players color palette list
        palette_interval = (0,5)                                                   # Color interval to extract from dominant colors palette (1rd to 5th color)

        ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
        for imag in imgs_list:
            # if int(j) == 0:
                # bbox = results_players[0].boxes.xyxy.cpu().numpy()[i,:]                         # Get bbox info (x,y,x,y)
                # obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]       # Crop bbox out of the frame
                imag = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB) 
                obj_img_w, obj_img_h = imag.shape[1], imag.shape[0]
                
                center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
                center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
                center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
                center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)

                center_filter = imag[center_filter_y1:center_filter_y2, center_filter_x1:center_filter_x2]
                
                obj_pil_img = Image.fromarray(np.uint8(center_filter))                          # Convert to pillow image
                reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)                   # Convert to web palette (216 colors)
                palette = reduced.getpalette()                                                  # Get palette as [r,g,b,r,g,b,...]
                palette = [palette[3*n:3*n+3] for n in range(256)]                              # Group 3 by 3 = [[r,g,b],[r,g,b],...]
                color_count = [(n, palette[m]) for n,m in reduced.getcolors()]                  # Create list of palette colors with their frequency
                RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(       # Create dataframe based on defined palette interval
                                    by = 'cnt', ascending = False).iloc[
                                        palette_interval[0]:palette_interval[1],:]
                palette = list(RGB_df.RGB)                                                      # Convert palette to list (for faster processing)
                
                # Update detected players color palette list
                obj_palette_list.append(palette)
        
        ## Calculate distances between each color from every detected player color palette and the predefined teams colors
        players_distance_features = []
        # Loop over detected players extracted color palettes
        for palette in obj_palette_list:
            palette_distance = []
            palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]  # Convert colors to L*a*b* space
            # Loop over colors in palette
            for color in palette_lab:
                distance_list = []
                # Loop over predefined list of teams colors
                for c in color_list_lab:
                    #distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                    distance = skimage.color.deltaE_cie76(color, c)                             # Calculate Euclidean distance in Lab color space
                    distance_list.append(distance)                                              # Update distance list for current color
                palette_distance.append(distance_list)                                          # Update distance list for current palette
            players_distance_features.append(palette_distance)                                  # Update distance features list

        ## Predict detected players teams based on distance features
        players_teams_list = []
        
        # Loop over players distance features
        for distance_feats in players_distance_features:
            vote_list=[]
            # Loop over distances for each color 
            for dist_list in distance_feats:
                team_idx = dist_list.index(min(dist_list))//nbr_team_colors                     # Assign team index for current color based on min distance
                vote_list.append(team_idx)                                                      # Update vote voting list with current color team prediction
            players_teams_list.append(max(vote_list, key=vote_list.count))                      # Predict current player team by vote counting


        return players_teams_list

@st.cache_resource
def grid_image(detections_imgs_list):
    detections_imgs_grid = []
    padding_img = np.ones((80,60,3),dtype=np.uint8)*255

    
    detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2)])
    detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2, len(detections_imgs_list))])

    if len(detections_imgs_list)%2 != 0:
        detections_imgs_grid[0].append(padding_img)
    #     if len(detections_imgs_list)%2 != 0:
    #         detections_imgs_grid.append(padding_img)


    concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
    concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
    concat_det_imgs = cv2.vconcat([concat_det_imgs_row1,concat_det_imgs_row2])
    concat_det_imgs = cv2.cvtColor(concat_det_imgs, cv2.COLOR_BGR2RGB)
    return concat_det_imgs

@st.cache_resource
def get_pixel_color(image, x, y):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get the color of the specified pixel
    pixel_color_hsv = hsv_image[y, x]
    
    # Convert HSV color to BGR
    pixel_color_bgr = cv2.cvtColor(np.array([[pixel_color_hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    
    return pixel_color_bgr

@st.cache_resource
def create_colors_info(team1_name, team1_shirt_color_rgb, team1_short_color_rgb, team1_socks_color_rgb, team1_gk_color_rgb, team2_name, team2_shirt_color_rgb, team2_short_color_rgb, team2_socks_color_rgb, team2_gk_color_rgb):
    team1_shirt_color_rgb = ImageColor.getcolor(team1_shirt_color_rgb, "RGB")
    team1_short_color_rgb = ImageColor.getcolor(team1_short_color_rgb, "RGB")
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color_rgb, "RGB")
    team1_socks_color_rgb = ImageColor.getcolor(team1_socks_color_rgb, "RGB")

    team2_shirt_color_rgb = ImageColor.getcolor(team2_shirt_color_rgb, "RGB")
    team2_short_color_rgb = ImageColor.getcolor(team2_short_color_rgb, "RGB")
    team2_socks_color_rgb = ImageColor.getcolor(team2_socks_color_rgb, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color_rgb, "RGB")

    colors_dic = {
        team1_name:[ team1_shirt_color_rgb, team1_short_color_rgb, team1_socks_color_rgb, team1_gk_color_rgb],
        team2_name:[team2_shirt_color_rgb , team2_short_color_rgb, team2_socks_color_rgb, team2_gk_color_rgb]
    }
    colors_list = colors_dic[team1_name]+colors_dic[team2_name] # Define color list to be used for detected player team prediction
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] # Converting color_list to L*a*b* space
    return colors_dic, color_list_lab

def create_or_check_folder(parent_directory, folder_name):
    # Join the parent directory and folder name to create the full path
    folder_path = os.path.join(parent_directory, folder_name)
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create the folder
        os.makedirs(folder_path)
    
    # Return the full path of the folder
    return folder_path