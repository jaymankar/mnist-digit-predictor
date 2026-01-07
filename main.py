from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from model.model_load import MyModel
from Image_preprocessing.preprocessing import preprocess_image

app = FastAPI()
class ImageRequest(BaseModel):
    image:str
    
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_methods=["*"],
    allow_headers=["*"]
)    

device = torch.device("cpu")
model = MyModel(input=1).to(device)  
state_dict = torch.load("model/model.pth",map_location=device)
model.load_state_dict(state_dict)
model.eval()



@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()
    
    
    
    

@app.post('/predict')
def prediction(data:ImageRequest):
    
    Img = data.image
    x = preprocess_image(Img)
    
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
    return {
        "prediction": pred.item(),
        "probabilities": probs.squeeze().tolist()
    }
    
    
    
    
    
    
    
    
    