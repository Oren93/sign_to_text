from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from pathlib import Path
from predict import make_prediction

UPLOAD_DIR = Path("../uploads")

app = FastAPI()

@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    path = UPLOAD_DIR / file_upload.filename
    with open(path, "wb") as f:
        f.write(data)
    make_prediction(path)


    ## Do something to the file
    
    return "short"
'''

'''