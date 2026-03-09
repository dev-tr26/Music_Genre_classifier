import torch
from pathlib import Path

def load_model():
    model_path = r"..\model_1\music_genre_model_1.pt"
    model = torch.jit.load(model_path, map_location="cpu")
    model = torch.jit.optimize_for_inference(model)
    
    model.eval()
    return model