from fastapi import FastAPI, UploadFile
from pathlib import Path

UPLOAD_DIR = Path() / "uploads"

app = FastAPI()

@app.post('/uploadfile/')
async def create_upload_file(file_upload: UploadFile):
    data = await file_upload.read()
    save_to = UPLOAD_DIR / file_upload.filename
    with open(save_to, "wb") as f:
        f.write(data)

    ## Do something to the file
    
    
    return {"filename": file_upload.filename}

