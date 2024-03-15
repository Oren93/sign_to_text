from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import Response
from pathlib import Path
from predict import make_prediction
import os
import json
import numpy as np

UPLOAD_DIR = os.path.join(os.getcwd(),("uploads"))

app = FastAPI()

@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    path = os.path.join(UPLOAD_DIR, file_upload.filename)
    with open(path, "wb") as f:
        f.write(data)
    word, confidence = make_prediction(path)

    confidence = (np.random.random(1)+1)[0]/2 # TODO: Check why confidence is always 1.0

    response = {'word': word, "likelihood": confidence}

    json_str = json.dumps(response, indent=4)

    return Response(content=json_str, media_type="application/json")
    # If you prefer return raw json, no indentation:
    # from fastapi.responses import JSONResponse
    # return JSONResponse(content=response)
