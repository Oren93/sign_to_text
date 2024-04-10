from fastapi import FastAPI, UploadFile, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json

import os
from starlette.requests import Request

#from predict import make_prediction # Temporarily commented for faster server reload

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
############################################
@app.post("/live/stream")
async def receive_video(video: UploadFile = File(...)):
    print("video.filename: ", video.filename)
    with open(video.filename, "wb") as f:
        f.write(await video.read())
    print("f", f)

    # Open the video file using a video library
    cap = cv2.VideoCapture(video.filename)
    
    # Get the number of frames in the video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("total_frames:", total_frames)
    i = 0
    # Iterate through the frames and print the frame number
    # for i in range(int(total_frames)): # Currently fails, total_frames is -2e17 for some reason
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        do_something_to_frame(frame) # To be replaced by a real code
        i += 1
        if i % 100 == 0:
            print(f"Frame no {i} and still running")
    print(f"Frames: {i}")
        
    return {"message": "Video received"}
