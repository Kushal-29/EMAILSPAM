"""
Data preprocessing module for email spam detection
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class EmailPreprocessor:
    """Class for preprocessing email text data"""
    
    def __init__(self):
        """Initialize preprocessor with stemmer and stopwords"""
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text):
        """
        Tokenize text and apply stemming
        
        Args:
            text (str): Input text
            
        Returns:
            str: Processed text with stemmed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        filtered_tokens = []
        for word in tokens:
            if word not in self.stop_words and len(word) > 2:
                stemmed_word = self.stemmer.stem(word)
                filtered_tokens.append(stemmed_word)
        
        return ' '.join(filtered_tokens)
    
    def preprocess_dataframe(self, df):
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe with 'text' column
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df = df.copy()
        
        # Convert labels to binary
        if 'label' in df.columns:
            df['label'] = df['label'].map({'ham': 0, 'spam': 1, 0: 0, 1: 1})
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Tokenize and stem
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_stem)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        return df
    
    def preprocess_single(self, text):
        """
        Preprocess single text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_stem(cleaned)
        return processed

# For testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = EmailPreprocessor()
    
    test_text = "Hello World! This is a TEST email 123."
    cleaned = preprocessor.clean_text(test_text)
    processed = preprocessor.tokenize_and_stem(cleaned)
    
    print("Test Preprocessor:")
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Processed: {processed}")