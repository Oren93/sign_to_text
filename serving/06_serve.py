# %% imports
import os
import tensorflow.keras as keras
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel  

# %%
MODEL_DIR = 'lstm'
PORT = 8001

def load_model(version=0):
    version = str(version)
    '''
    Load, by default, the newest TF model, unless a specified version is requested
    '''
    models = [f.name for f in os.scandir(MODEL_DIR) if f.is_dir()]
    if version == '0':
        version = max(models)
    else:
        if version not in models:
            return 0 # TODO: error handling
    model_path = os.path.join(MODEL_DIR,version,'sign_to_text.keras')
    model = keras.models.load_model(model_path)
    return model
# %%
app = FastAPI()

class Video(BaseModel):
    video: UploadFile

@app.post('/predict')
async def predict(video: UploadFile = File(...)):
    model = load_model()
    if model == 0:
        return {"error: model not found"}
    # Example: Save the uploaded file to a local directory (replace this with your actual processing logic)
    with open(f"uploaded_video_{video.filename}", "wb") as f:
        f.write(video.file.read())
        
    result = {"prediction": "Your video classification result"}
    
    return result

@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <html>
        <head>
            <title>Tensorflow-server</title>
        </head>
        <body style="background-color:black;color:white;">
            <h1>FastAPI</h1>
            <p>This is a FastAPI server to serve a TensorFlow model. For trying yourself, go to</p>
            <p>http://localhost:{PORT}/docs#/</p>
        </body>
    </html>
    """

#%% TRYOUT FROM TUTORIAL
from typing import Optional
from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()


templates = Jinja2Templates(directory="templates")


@app.get("/index/", response_class=HTMLResponse)
async def movielist(request: Request, hx_request: Optional[str] = Header(None)):
    films = [
        {'name': 'Blade Runner', 'director': 'Ridley Scott'},
        {'name': 'Pulp Fiction', 'director': 'Quentin Tarantino'},
        {'name': 'Mulholland Drive', 'director': 'David Lynch'},
    ]
    context = {"request": request, 'films': films}
    if hx_request:
        return templates.TemplateResponse("partials/table.html", context)
    return templates.TemplateResponse("index.html", context)   
##################

# For debugging:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)