import datetime
from fastapi import FastAPI, Form, UploadFile, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json

import os
from starlette.requests import Request

#from predict import make_prediction # Temporarily commented for faster server reload
from live_predict import make_live_prediction

UPLOAD_DIR = os.path.join(os.getcwd(),("uploads"))

app = FastAPI()
origins = [
    "http://localhost:8080",  # Update this with the origin of your frontend application
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
    # The following dummy code is for faster server reload to prevent loading the model
    return   json.dumps(
         {"filename": 'dummy.mp4',
          "predicted_word": 'dummy',
          "confidence_level": str(0.77)}
    )
    predicted_word, confidence = make_prediction(path) # TODO: Check why confidence is always 1.0

    ret = {"filename": file_upload.filename, "predicted_word": predicted_word, "confidence_level": str(confidence)}
    return json.dumps(ret)


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import cv2

app = FastAPI()

# Allow all origins for CORS (adjust this as needed for your deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
###########################################
# Dummy, should replace with serious frame handler,
# currently only validating that the video is being sent properly
def do_something_to_frame(frame):
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
last_proccessed_frame = 1 # TODO: find a solid solution, this is a temporary solution
chunk_number = 0
############################################
@app.post("/live/stream")
async def receive_video(video: UploadFile = File(...), rate: int = Form(...)):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join('uploads',f"live_{timestamp}.mp4")
    global last_proccessed_frame
    global chunk_number 

    with open(filename, "wb") as f:
        f.write(await video.read())
    print("video.filename, filename: ", video.filename,filename)        
    print("rate: ", rate)
    
    # Open the video file using a video library
    cap = cv2.VideoCapture(filename)
    
    # Get the number of frames in the video. TODO: Check why it is corrupted
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("total_frames:", total_frames)
    
    last_proccessed_frame, predictions = make_live_prediction(filename,rate,last_proccessed_frame)
    predictions = predictions.tolist()
    chunk_number += 1
    return {"chunk":chunk_number,
            "words": predictions}

@app.post('/live/stop')
async def stop_recording():
    global last_proccessed_frame
    global chunk_number
    print(last_proccessed_frame)
    last_proccessed_frame = 0
    chunk_number = 0
    print(last_proccessed_frame)
    return {"message": "Video stopped"}


