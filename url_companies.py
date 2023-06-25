"""
    This script works making predictions and identifying 
known companies inside urls and instagram usernames.

Necessary archives:
    - Model: text_model.pkl
    - Vectorizer: text_vectorizer.pkl
"""
import os
import re
import joblib
import pandas as pd


class UrlsCompanies:
    def __init__(self):
        """
            Initialize class, loads Neural Network model 
        and vectorizer (TfidfVectorizer) used on training
        """
        self.model_path = os.path.dirname(os.path.abspath(__file__))
        self.model = joblib.load(
            open(os.path.join(self.model_path, 'text_model.pkl'), 'rb'))
        self.vectorizer = joblib.load(
            open(os.path.join(self.model_path, 'text_vectorizer.pkl'), 'rb'))

    def clean_url(self, url) -> str:
        """
        Takes a url in string format and returns it clean and ready for predictions
        """
        cleaned_url = re.sub('[/:.?=#$%~@!-_+]', ' ', url)
        return cleaned_url

    def predict_url(self, url) -> str:
        """
        Takes a string and returns the prediction
        """
        cleaned_url = self.clean_url(url)
        vec_urls = self.vectorizer.transform([cleaned_url])
        prediction = self.model.predict(vec_urls)[0]
        return prediction

    