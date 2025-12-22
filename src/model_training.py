"""
Model training and evaluation module for email spam detection
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SpamClassifier:
    """Class for training and evaluating spam classifiers"""
    
    def __init__(self, model_type='naive_bayes'):
        """
        Initialize classifier
        
        Args:
            model_type (str): Type of model ('naive_bayes' or 'svm')
        """
        self.model_type = model_type
        
        if model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True, C=1.0, random_state=42)
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            array: Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            array: Prediction probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For SVM without probability, return 0/1
            predictions = self.predict(X)
            probas = np.zeros((len(predictions), 2))
            for i, pred in enumerate(predictions):
                probas[i, pred] = 1.0
            return probas
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            tuple: (metrics dict, confusion matrix, classification report)
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
        
        return metrics, cm, report
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Ham', 'Spam'],
                   yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_model(self, path):
        """
        Save trained model to disk
        
        Args:
            path (str): Path to save model
        """
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """
        Load model from disk
        
        Args:
            path (str): Path to load model from
        """
        self.model = joblib.load(path)

# For testing
if __name__ == "__main__":
    print("SpamClassifier module loaded successfully!")