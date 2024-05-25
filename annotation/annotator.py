import cv2
import supervision as sv
import numpy as np
from .detection_class import Detections

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
corner_annotator = sv.BoxCornerAnnotator()
ellipse_annotator = sv.EllipseAnnotator()
heat_map_annotator = sv.HeatMapAnnotator()
trace_annotator = sv.TraceAnnotator()
triangle_annotator = sv.TriangleAnnotator()
tracker = sv.ByteTrack()

def crop_detections(image, df):
    if type(image) == str:
        img = cv2.imread(image)
    else:
        img = image
    football = df[df['class_id']==0]
    others = df[df['class_id']!=0]

    if football.shape[0] != 0:
        football_detections = Detections.from_dataframe(football)
        img = triangle_annotator.annotate( scene=img, detections=football_detections)

    others_detections = Detections.from_dataframe(others)
    detections_imgs_list = []
    bboxes = others_detections.xyxy

    for i in range(len(bboxes)):
            bbox = bboxes[i]                         
            obj_img = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            obj_img = cv2.resize(obj_img, (60,80))
            detections_imgs_list.append(obj_img)
    return detections_imgs_list

def annotate_frame(image, df):
    if type(image) == str:
        img = cv2.imread(image)
    else:
        img = image

    football = df[df['class_id']==0]
    others = df[df['class_id']!=0]

    if football.shape[0] != 0:
        football_detections = Detections.from_dataframe(football)
        img = triangle_annotator.annotate( scene=img, detections=football_detections)

    others_detections = Detections.from_dataframe(others)
    label = [ f'Id {object_id}' for object_id in others_detections.tracker_id]
    annotated_image = ellipse_annotator.annotate( scene=img, detections=others_detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=others_detections, labels=label)

    return annotated_image

def annotate(video_file, df, start_frame=None, end_frame=None, file_name='./video/annotated_out.mp4'):

    if type(video_file) == str:
        video = cv2.VideoCapture(video_file)
    else:
        video = video_file   

    starting_frame = 1 if start_frame is None else start_frame
    ending_fame = df['frame'].max() if end_frame is None else end_frame

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter( file_name, fourcc, fps, (width, height))
    count = 1
    for _ in range(ending_fame):
        ret, frame = video.read()
        if ret:
            if (count <= ending_fame) & (count >= starting_frame):
                detection_df = df[df['frame']==_]
                if detection_df.shape[0] != 0:
                    annotated_image = annotate_frame(frame.copy(), detection_df)
                else:
                    annotated_image = frame.copy()
                out.write(annotated_image)
            elif count > ending_fame:
                break
            count = count + 1


        else:
            break
    video.release()
    out.release()
