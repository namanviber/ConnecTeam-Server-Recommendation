import os

class Config:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_PATH, 'data')
    MODEL_PATH = os.path.join(BASE_PATH, 'model', 'recommendation_model.pkl')
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)