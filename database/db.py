from typing import Optional, List, Dict, Union
# One line of FastAPI imports here later 👈
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship, DateTime
import pandas as pd
import numpy as np
import streamlit as st

sqlite_file_name = "./database/soka_db_00.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=True,)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

class Event(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    club: str = Field(index=True)
    name: str = Field(index=True)
    half: str = Field(index=True)

    start_min: int
    start_sec: int

    end_min: int
    end_sec: int

    match_id: Optional[int] = Field(default=None, foreign_key="match.id")

class Detection(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)

    frame: int 

    object_id: int = Field(default=None, index=True)
    class_id: int = Field(index=True)
    class_name: str = Field(index=True)

    conf_score: float
    x1: float
    x2: float
    y1: float
    y2: float

    mask: str = Field(default=None)
    model: str = Field(default=None)

    # club: str = Field(default=None)
    player_id: Optional[int] = Field(default=None, foreign_key="player.id") 
    video_id: Optional[int] = Field(default=None, foreign_key="match.id")

class Player(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    club: str
    position: Optional[str] = Field(default=None)
    jersey: Optional[int] = Field(default=None)
    # detection_id: Optional[Detection] = Field(default=None)

class Corners(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)

    vid_name: str
    frame: int

    x: int
    y: int
    point: int

class Match(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)

    video: str

    teamA: str
    teamB: str
    stadium: str
    date: str
 
    teamA_Colors: Optional[str] = Field(default=np.nan)
    teamB_Colors: Optional[str] = Field(default=np.nan)

class MatchDetail(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    moment_video: str
    club:str
    player: str

    event_type: str

    frame: int

    moment_id: Optional[int] = Field(default=None, foreign_key="event.id") 
    match_id: Optional[int] = Field(default=None, foreign_key="match.id")

class PlayerDetection(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    club: str = Field(index=True)
    detection_id: Optional[int] = Field(default=None, foreign_key="detection.id")


event_type = ['Goal', 'Yellow Card', 'Red Card', 'Substitute', 'Foul']

shot = ['Shot of target', 'Shot on target', 'Shot Blocked', 'Shot from the box', 'Shot outside the box']
attacking_actions = ['attacks with shot', 'Counter-Attacks', 'Counter-Attacks with shot', 'dribble']
set_pieces = ['Corner', 'Throw In', 'Free Kick', 'Goal Kick']
pass_events = ['Pass (10-25m)', 'Key Pass', 'Long Pass (>25m)', 'Short Pass (0-10m)', 'Progressive Pass', 'Back Pass', 'Cross']
defensive_events = ['Clearnace', 'Block', 'Interception', 'Disposition']
aerial_actions = ['Header', 'Aerial Duel']


maps = ['Goal Map', 'Shot Map', 'Average Position Heat Map', 'Key Pass Map', 'Pass Map', 'Ball Lost Map']
########### DETECTION ###########

def save_detection(video_id, frame_number,  detection_result, moment_id=0, model='yolov9'):

    obj_classes=['Football', 'Goalkeeper', 'Outfield_Player', 'Referee', 'Side_Ref']

    if detection_result is not None:
        with Session(engine) as session:    
            for i in range(len(detection_result)):

                xyxy = detection_result.xyxy[i].tolist()
                tracker_id = int(detection_result.tracker_id[i]) if detection_result.tracker_id is not None else None
                class_id = int(detection_result.class_id[i])
                confidence = detection_result.confidence[i]

                detection = Detection(  video_id = video_id, 
                                        frame = frame_number, 
                                        object_id = tracker_id, 
                                        class_id = class_id, 
                                        class_name = obj_classes[class_id], 
                                        conf_score = confidence , 
                                        x1 = xyxy[0], 
                                        x2 = xyxy[2], 
                                        y1 = xyxy[1], 
                                        y2 = xyxy[3],
                                        mask='mask',
                                        model=model
                                        )
                
                session.add(detection)

                session.commit()
    else:
        pass


def update_detections(id, tracker_id, player=None):
    with Session(engine) as session:
        statement = select(Detection).where(Detection.id == id)
        results = session.exec(statement)
        # print('Query results ******* ', results)
        try:
            detection = results.one()
          
            detection.player_id = player
            detection.object_id = tracker_id
 

            # Committing and refreshing database
            session.add(detection)
            session.commit()
            session.refresh(detection)


        except IndexError:
            print('passed ', id)


def delete_row(id):
    with Session(engine) as session:
        statement = select(Detection).where(Detection.id == id)
        results = session.exec(statement)
        detection = results.one()
        print("Detection: ", detection)
########### EVENT ###########

def save_moment(club, half, moment_name, start_min, start_sec, end_min, end_sec, video_id):
    with Session(engine) as session: 
            event = Event(  club= club, 
                            half =half,
                            name= moment_name, 
                            start_min=start_min,
                            start_sec = start_sec, 
                            end_min = end_min,
                            end_sec = end_sec, 
                            match_id = video_id )

            session.add(event)
            session.commit()
            session.refresh(event)

########### MATCH ###########
def read_match(vid_name):
    with Session(engine) as session:
        statement = select(Match).where(Match.video == vid_name)
        results = session.exec(statement).all()
        df = query_to_df(results)
        return df
    
def update_match(id, teamA_colors, teamB_colors):

    with Session(engine) as session:
        statement = select(Match).where(Match.id == id)
        results = session.exec(statement)
        # print('Query results ******* ', results)
        try:
            match_info = results.one()
            # Update fields            
            match_info.teamA_Colors = teamA_colors
            match_info.teamB_Colors = teamB_colors

            # Committing and refreshing database
            session.add(match_info)
            session.commit()
            session.refresh(match_info)


        except IndexError:
            print('passed ', id)  

def save_match(video, teamA, teamB, stadium, date):
    with Session(engine) as session: 
            match_info = Match( video= video, 
                            teamA=teamA,
                            teamB = teamB,
                            stadium = stadium,
                            date = date
                            )

            session.add(match_info)
            session.commit()
            session.refresh(match_info)   
########### PLAYER ###########

def save_players(club, name):
    with Session(engine) as session:
            player_info = Player( club= club, name=name)             
            session.add(player_info)
            session.commit()
            session.refresh(player_info)  

########### CORNER ###########
def save_corners(name, x, y, p, frame):
    corner = Corners(vid_name=name, x=x, y=y, point=p, frame=frame)
    with Session(engine) as session:
            session.add(corner)
            session.commit()
            session.refresh(corner)

########### READ DATABASE TABLES ###########

def query_to_df(results):
    data = []
    for row in results:
        data.append(row.model_dump())

    df = pd.DataFrame(data)

    return df


def read_table(name):
    with Session(engine) as session: 
        query = select(name)
        results = session.exec(query).all()
        df = query_to_df(results)
        return df

def read_db(video_name):
    with Session(engine) as session: 
        query = select(Detection).where(Detection.video == video_name)
        # query = select(Detection)
        results = session.exec(query).all()
   
    # Iterate through the results
        data = []
        for row in results:
            data.append(row.model_dump())

        df = pd.DataFrame(data)

        return df


def read_all():
    with Session(engine) as session: 
        query = select(Detection)
        # query = select(Detection)
        results = session.exec(query).all()
   
    # Iterate through the results
        data = []
        for row in results:
            data.append(row.dict())

        df = pd.DataFrame(data)

        return df
            

########### Match Details ###########

def save_moment_detail(moment_video, club, Player, event_type, frame, moment_id, match_id):
    with Session(engine) as session: 
            event_detail = MatchDetail(  moment_video= moment_video, 
                            club =club,
                            Player= Player, 
                            event_type=event_type,
                            frame = frame, 
                            moment_id = moment_id,
                            match_id = match_id )

            session.add(event_detail)
            session.commit()
            session.refresh(event_detail)

########### CREAT DB and TABLES ###########
# create_db_and_tables()