import gensim.downloader as api
from gensim.models import KeyedVectors
import os

def load_fasttext_model():
    model_path = 'fasttext/fasttext_model.bin'
    
    os.makedirs('fasttext', exist_ok=True)
    
    if not os.path.exists(model_path):
        print("Model not found. Downloading FastText model...")
        
        model = api.load('fasttext-wiki-news-subwords-300')
        
        model.save(model_path)
        print("Model downloaded and saved successfully!")
    else:
        print("Loading FastText model from cache...")
        model = KeyedVectors.load(model_path)
        print("Model loaded successfully!")
    
    return model