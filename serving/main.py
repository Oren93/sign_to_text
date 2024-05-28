import datetime
from fastapi import FastAPI, Form, UploadFile, Request, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import json
import cv2
import os

from predict import make_prediction
from live_predict import make_live_prediction

UPLOAD_DIR = os.path.join(os.getcwd(),("uploads"))

app = FastAPI()
origins = [
    "http://localhost:8080",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile, request: Request):
    data = await file_upload.read()
    path = os.path.join(UPLOAD_DIR, file_upload.filename)
    with open(path, "wb") as f:
        f.write(data)

    predicted_word, confidence = make_prediction(path)

    ret = {"filename": file_upload.filename, "predicted_word": predicted_word, "confidence_level": str(confidence)}
    return json.dumps(ret)

#%% TODO: find a solid solution involving the frontend, this is a temporary solution
last_proccessed_frame = 1
chunk_number = 0
#%%
@app.post("/live/stream")
async def receive_video(video: UploadFile = File(...), rate: int = Form(...)):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join('uploads',f"live_{timestamp}.mp4")
    global last_proccessed_frame
    global chunk_number 

    with open(filename, "wb") as f:
        f.write(await video.read())
    
    # Open the video file using a video library
    cap = cv2.VideoCapture(filename)
    
    # Get the number of frames in the video. TODO: Check why it is corrupted
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    last_proccessed_frame, predictions = make_live_prediction(filename,rate,last_proccessed_frame)
    predictions = predictions.tolist()
    chunk_number += 1
    return {"chunk":chunk_number,
            "words": predictions}

@app.post('/live/stop')
async def stop_recording():
    global last_proccessed_frame
    global chunk_number
    last_proccessed_frame = 0
    chunk_number = 0
    return {"message": "Video stopped"}
