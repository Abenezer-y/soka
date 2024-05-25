import cv2
import datetime
import numpy as np
import streamlit as st


@st.cache_resource
def convert_to_frame_count(fps, min, sec):
    sec_in_min = min*60
    total_sec = sec + sec_in_min
    frame_count = total_sec * fps
    return frame_count

@st.cache_resource
def video_meta(video_path):
    cap = cv2.VideoCapture(video_path) 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, frames_count

# Function to add timestamp to frame
@st.cache_resource
def add_timestamp(frame, elapsed_time):
    timestamp_text = str(elapsed_time)
    cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

@st.cache_resource
def timestamp( video_path, output_path):
    # Video capture

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    elapsed_seconds = 0
    frame_count = 0
    # Process frames
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count == 0 or frame_count % fps == 0:
            # Your code here
            elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
            elapsed_seconds = elapsed_seconds + 1
            # Add your logic or function calls here
        else:
            pass
        

        # Add timestamp to the frame
        frame_with_timestamp = add_timestamp(frame, elapsed_time_str)

        # Write the frame with timestamp to the output video
        out.write(frame_with_timestamp)
        frame_count = frame_count + 1

    # Release video capture and writer
    cap.release()
    out.release()

@st.cache_resource
def histogram_equalization(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized_frame = cv2.equalizeHist(gray_frame)
    
    # Convert back to BGR (if needed)
    equalized_frame_bgr = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)
    
    return equalized_frame_bgr

@st.cache_resource
def normalize_frame(frame):
    # Normalize pixel values to the range [0, 1]
    normalized_frame = frame.astype(np.float32) / 255.0
    
    return normalized_frame

@st.cache_resource
def reduce_noise(frame, kernel_size=(5, 5)):
    # Apply a Gaussian blur for noise reduction
    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)
    
    return blurred_frame

@st.cache_resource
def pad_frame(frame, padding_size=50):
    # Pad the frame with a specified size
    padded_frame = cv2.copyMakeBorder(frame, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # print('padded x 50')
    return padded_frame

@st.cache_resource
def crop_frame(video_path, output_path, width, height, w1, w2, h1, h2):
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = cv2.imread(frame)[w1:w2, h1:h2]
        # Write the frame with timestamp to the output video
        out.write(cropped_frame)
    # Release video capture and writer
    cap.release()
    out.release()
    print('completed!!')

@st.cache_resource
def extract_frame(video_path, start, end, out_dir):
    cap = cv2.VideoCapture(video_path)
    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (i >= start) & (i < end):
            cv2.imwrite(f'{out_dir}/frame_{i}.jpeg', frame)
        elif i > end:
            break
        i = i+1
    cap.release()

@st.cache_resource
def trim_video(video_path, start_min, start_sec, end_min, end_sec, out_file_name='/Users/abenezer/Documents/Projects/SOKA/video_analyzer/functions/annotation/moment_out.mp4'):
    if type(video_path) == str:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = video_path
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = convert_to_frame_count(fps, start_min, start_sec)
    end = convert_to_frame_count(fps, end_min, end_sec)
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter(out_file_name, fourcc, fps, (width, height))
    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (i >= start) & (i < end):
            cv2.putText(frame, str(i), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame)
        elif i > end:
            break
        i = i+1
    cap.release()
    out.release()

@st.cache_resource
def process_frame(frame, rotate=False, crop=False, crop_coordinates=None, rotation_angle=None):
    processed_frame = frame.copy()

    if rotate:
        if rotation_angle is None:
            raise ValueError("Rotation angle must be specified.")
        
        # Rotate the frame
        height, width = frame.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)
        processed_frame = cv2.warpAffine(processed_frame, rotation_matrix, (width, height))

    if crop:
        if crop_coordinates is None:
            raise ValueError("Crop coordinates must be specified.")

        # Crop the frame
        x, y, w, h = crop_coordinates
        processed_frame = processed_frame[y:y+h, x:x+w]

    return processed_frame