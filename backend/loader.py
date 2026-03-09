import torch
from pathlib import Path
import gdown

MODEL_DIR = Path("models")
MODEL_PATH = r"..\model_1\music_genre_model_1.pt"


FILE_ID = "10sUE7cKe7pk9Knj4Smymv16CbTxKZgKJ"

def download_model():
    MODEL_DIR.mkdir(exist_ok=True)

    url = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=drive_link"

    print("Downloading model from Google Drive...")
    gdown.download(url, str(MODEL_PATH), quiet=False)



def load_model():
    
    if not MODEL_PATH.exists():
        download_model()
        
    
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model = torch.jit.optimize_for_inference(model)
    
    model.eval()
    return model