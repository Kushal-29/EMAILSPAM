"""
Feature engineering module for email spam detection
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

class FeatureEngineer:
    """Class for feature engineering using TF-IDF"""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize feature engineer
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams to use
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range,
            max_df=0.7,
            min_df=5
        )
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts
        
        Args:
            texts (iterable): List of text documents
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        X = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return X
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts (iterable): List of text documents
            
        Returns:
            scipy.sparse matrix: TF-IDF features
            
        Raises:
            ValueError: If vectorizer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        return self.vectorizer.transform(texts)
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size (float): Proportion of test data
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    
    def save_vectorizer(self, path='models/vectorizer.pkl'):
        """
        Save fitted vectorizer to disk
        
        Args:
            path (str): Path to save vectorizer
        """
        joblib.dump(self.vectorizer, path)
    
    def load_vectorizer(self, path='models/vectorizer.pkl'):
        """
        Load vectorizer from disk
        
        Args:
            path (str): Path to load vectorizer from
        """
        self.vectorizer = joblib.load(path)
        self.is_fitted = True

# For testing
if __name__ == "__main__":
    print("FeatureEngineering module loaded successfully!")