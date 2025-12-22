"""
Inference module for making predictions with trained models
"""
import joblib
import numpy as np
import os
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import EmailPreprocessor

class SpamDetector:
    """Class for making predictions with trained spam detector"""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initialize spam detector
        
        Args:
            model_path (str): Path to trained model
            vectorizer_path (str): Path to vectorizer
        """
        self.preprocessor = EmailPreprocessor()
        
        # Set default paths if not provided
        if model_path is None:
            model_path = 'models/naive_bayes_model.pkl'
        if vectorizer_path is None:
            vectorizer_path = 'models/vectorizer.pkl'
        
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # Load model and vectorizer
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.model_loaded = True
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Please train the model first by running train.py")
            self.model_loaded = False
    
    def predict(self, email_text):
        """
        Predict if an email is spam
        
        Args:
            email_text (str): Email text
            
        Returns:
            dict: Prediction results
        """
        if not self.model_loaded:
            return {
                'error': 'Model not loaded. Please train the model first.',
                'prediction': 'unknown'
            }
        
        # Preprocess the text
        processed_text = self.preprocessor.preprocess_single(email_text)
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        
        # Get probabilities
        try:
            probability = self.model.predict_proba(text_vectorized)[0]
            spam_prob = float(probability[1])
            ham_prob = float(probability[0])
        except:
            # If model doesn't have predict_proba
            spam_prob = 1.0 if prediction == 1 else 0.0
            ham_prob = 1.0 - spam_prob
        
        # Determine confidence
        confidence_level = 'HIGH' if abs(spam_prob - 0.5) > 0.3 else 'MEDIUM' if abs(spam_prob - 0.5) > 0.1 else 'LOW'
        
        return {
            'text': email_text[:100] + '...' if len(email_text) > 100 else email_text,
            'processed_text': processed_text,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'confidence': confidence_level
        }
    
    def batch_predict(self, email_texts):
        """
        Predict multiple emails
        
        Args:
            email_texts (list): List of email texts
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in email_texts:
            results.append(self.predict(text))
        return results

# For testing
if __name__ == "__main__":
    detector = SpamDetector()
    if detector.model_loaded:
        test_email = "Congratulations! You've won a free iPhone. Click here to claim!"
        result = detector.predict(test_email)
        print("Test prediction:")
        print(f"Email: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")
    else:
        print("Model not loaded. Please run train.py first.")