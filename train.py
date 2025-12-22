"""
Main training script for email spam detection
"""
import pandas as pd
import numpy as np
import re
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import EmailPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import SpamClassifier
from sklearn.model_selection import train_test_split

def main():
    print("=" * 60)
    print("EMAIL SPAM DETECTOR - TRAINING")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📂 Loading dataset...")
    try:
        df = pd.read_csv('data/spam.csv', encoding='latin-1')
    except:
        print("Error: Could not load data/spam.csv")
        print("Please ensure the dataset is in the data folder")
        return
    
    # Check and rename columns
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    
    # Keep only needed columns
    cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    print(f"Dataset loaded: {len(df)} emails")
    print(f"Spam: {df['label'].value_counts().get('spam', 0)}")
    print(f"Ham: {df['label'].value_counts().get('ham', 0)}")
    
    # Step 2: Preprocess
    print("\n🔄 Preprocessing data...")
    preprocessor = EmailPreprocessor()
    df = preprocessor.preprocess_dataframe(df)
    
    # Step 3: Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Step 4: Feature engineering
    print("\n🔧 Creating features...")
    fe = FeatureEngineer(max_features=3000)
    X_train_tfidf = fe.fit_transform(X_train)
    X_test_tfidf = fe.transform(X_test)
    
    # Step 5: Train models
    models_to_train = ['naive_bayes', 'svm']
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}...")
        print('='*50)
        
        classifier = SpamClassifier(model_type=model_name)
        classifier.train(X_train_tfidf, y_train)
        
        # Evaluate
        metrics, cm, report = classifier.evaluate(X_test_tfidf, y_test)
        
        print(f"\n📊 Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        # Save model
        classifier.save_model(f'models/{model_name}_model.pkl')
        print(f"✅ Model saved to models/{model_name}_model.pkl")
    
    # Save vectorizer
    fe.save_vectorizer('models/vectorizer.pkl')
    print("\n✅ Vectorizer saved to models/vectorizer.pkl")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()