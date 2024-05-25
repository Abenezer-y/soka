from ultralytics import YOLO
from database.db import save_detection
import cv2
import supervision as sv
import streamlit as st


box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale=1)
corner_annotator = sv.BoxCornerAnnotator()
ellipse_annotator = sv.EllipseAnnotator()
heat_map_annotator = sv.HeatMapAnnotator()
trace_annotator = sv.TraceAnnotator()
triangle_annotator = sv.TriangleAnnotator()
tracker = sv.ByteTrack(50)


# function to loop over a video and run inference
@st.cache_resource
def run_inference(input_video, video_id, out_dir, model_path, conf=0.1, iou=0.5, start=None, end=None):
    model = YOLO(model_path)

    if type(input_video) == str:
        video = cv2.VideoCapture(input_video)

    # Collect input videos FPS, width, and height
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # tracker = sv.ByteTrack(fps)
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(out_dir, fourcc, fps, (width, height))
    if start is None and end is None:
        start = 1
        end = frames_count

    

    i = 1
    # Read each video of the frame, run inference, annotate frame and make an output video 
    while video.isOpened():
        # Read in video
        ret, frame = video.read()
        if ret:
            if (start<=i<=end):
                results = model.track(source=frame, device='mps', iou=iou, conf=conf)
                # collect results and ad trackers 
                detections = sv.Detections.from_ultralytics(results[0])
                
                # detections = tracker.update_with_detections(detections)

                labels = [ f"#{tracker_id} {results[0].names[class_id]}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id) ]

                # annotate frame
                football = detections[detections.class_id == 0]
                others = detections[detections.class_id != 0]

                if len(football) > 0:
                    frame = triangle_annotator.annotate( scene=frame, detections=football)

                annotated_image = ellipse_annotator.annotate( scene=frame, detections=others)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels) 

                # write to output video
                out.write(annotated_image)
                print('Here')
                # save detection results for future use
                save_detection(video_id=video_id, frame_number=i, detection_result=detections)  
                i = i+1
                print('also here')
            else:
                break
        else:
            # Break the loop if the end of the video is reached
            break

    video.release()
    out.release()

@st.cache_resource
def run_inferene_on_frame(model_path, frame, vid, frame_no, conf=0.1, iou=0.5):
    image = frame.copy()
    model = YOLO(model_path)
    results = model.track(source=frame, device='mps', iou=iou, conf=conf)
    detections = sv.Detections.from_ultralytics(results[0])

    labels = [ f"#{tracker_id} {results[0].names[class_id]}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id) ]
    
    # annotate frame
    football = detections[detections.class_id == 0]
    others = detections[detections.class_id != 0]

    if len(football) > 0:
        image = triangle_annotator.annotate( scene=image, detections=football)
        frame = triangle_annotator.annotate( scene=image, detections=football)

    annotated_image = ellipse_annotator.annotate( scene=image, detections=others)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels) 
    detections_imgs_list = []
    bboxes = others.xyxy

    for i in range(len(bboxes)):
            bbox = bboxes[i]                         
            obj_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            obj_img = cv2.resize(obj_img, (60,80))
            detections_imgs_list.append(obj_img)

    save_detection(video_id=vid, moment_id=0, frame_number=frame_no, detection_result=detections) 
    
    return annotated_image, detections, detections_imgs_list
