import torch
from pathlib import Path
import gdown

DOWNLOAD_MODEL_PATH = Path("models/music_genre_model_1.pt")
LOCAL_MODEL_PATH = Path("../model_1/music_genre_model_1.pt")


FILE_ID = "10sUE7cKe7pk9Knj4Smymv16CbTxKZgKJ"

def download_model():
    DOWNLOAD_MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)

    url = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=drive_link"

    print("Downloading model from Google Drive...")
    gdown.download(url, str(DOWNLOAD_MODEL_PATH), quiet=False)



def load_model():
    
    if LOCAL_MODEL_PATH.exists():
        model_path = LOCAL_MODEL_PATH
    else:
        if not DOWNLOAD_MODEL_PATH.exists():
            download_model()
        model_path = DOWNLOAD_MODEL_PATH
        
    model = torch.jit.load(model_path, map_location="cpu")
    model = torch.jit.optimize_for_inference(model)
    
    model.eval()
    return model