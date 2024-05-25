# from typing import Optional, List, Dict, Union
# # One line of FastAPI imports here later 👈
# from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship, DateTime


# class Event(SQLModel, table=True):
#     __table_args__ = {"extend_existing": True}

#     id: Optional[int] = Field(default=None, primary_key=True)
#     club: str = Field(index=True)
#     name: str = Field(index=True)
#     half: str = Field(index=True)

#     start_min: int
#     start_sec: int

#     end_min: int
#     end_sec: int

#     match_id: Optional[int] = Field(default=None, foreign_key="match.id")

# class Detection(SQLModel, table=True):
#     __table_args__ = {"extend_existing": True}
#     id: Optional[int] = Field(default=None, primary_key=True)

#     frame: int 

#     object_id: int = Field(default=None, index=True)
#     class_id: int = Field(index=True)
#     class_name: str = Field(index=True)

#     conf_score: float
#     x1: float
#     x2: float
#     y1: float
#     y2: float

#     mask: str = Field(default=None)
#     model: str = Field(default=None)

#     club: str = Field(default=None)
#     player_id: Optional[int] = Field(default=None, foreign_key="player.id") 
#     video_id: Optional[int] = Field(default=None, foreign_key="match.id")

# class Player(SQLModel, table=True):
#     __table_args__ = {"extend_existing": True}
#     id: Optional[int] = Field(default=None, primary_key=True)
#     name: str
#     club: str
#     position: Optional[str] = Field(default=None)
#     jersey: Optional[int] = Field(default=None)
#     # detection_id: Optional[Detection] = Field(default=None)

# class Corners(SQLModel, table=True):
#     __table_args__ = {"extend_existing": True}
#     id: Optional[int] = Field(default=None, primary_key=True)

#     vid_name: str
#     frame: int

#     x: int
#     y: int
#     point: int

# class Match(SQLModel, table=True):
#     __table_args__ = {"extend_existing": True}
#     id: Optional[int] = Field(default=None, primary_key=True)

#     video: str

#     teamA: str
#     teamB: str
#     stadium: str
#     date: str
 
#     teamA_Colors: Optional[str] = Field(default=np.nan)
#     teamB_Colors: Optional[str] = Field(default=np.nan)

# class MatchDetail(SQLModel, table=True):
#     __table_args__ = {"extend_existing": True}
#     id: Optional[int] = Field(default=None, primary_key=True)
#     moment_video: str
#     club:str
#     player: str

#     event_type: str

#     frame: int

#     moment_id: Optional[int] = Field(default=None, foreign_key="event.id") 
#     match_id: Optional[int] = Field(default=None, foreign_key="match.id")


