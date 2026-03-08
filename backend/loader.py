import torch

def load_model():
    model = torch.jit.load(r"C:\Desktop\github_projects\Music_genre_classifier\model_1\music_genre_model_1.pt")
    model.eval()
    return model