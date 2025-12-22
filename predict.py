import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import SpamDetector
import argparse

def main():
    parser = argparse.ArgumentParser(description='Predict if an email is spam')
    parser.add_argument('--email', type=str,
                       help='Email text to classify')
    parser.add_argument('--file', type=str,
                       help='File containing emails (one per line)')
    parser.add_argument('--model', type=str, default='naive_bayes',
                       choices=['naive_bayes', 'svm'],
                       help='Model to use for prediction')
    
    args = parser.parse_args()
    
    # Load detector
    model_path = f'models/{args.model}_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
    
    detector = SpamDetector(model_path=model_path)
    
    if args.email:
        # Single email prediction
        result = detector.predict(args.email)
        print(f"\nEmail: {result['text'][:100]}...")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")
        print(f"Ham Probability: {result['ham_probability']:.2%}")
        
    elif args.file:
        # Batch prediction from file
        with open(args.file, 'r') as f:
            emails = f.readlines()
        
        print(f"Analyzing {len(emails)} emails...\n")
        results = detector.batch_predict(emails)
        
        spam_count = sum(1 for r in results if r['prediction'] == 'spam')
        print(f"Results: {spam_count} spam, {len(results)-spam_count} ham")
        
        for i, result in enumerate(results[:5]):  # Show first 5
            print(f"\n{i+1}. {result['text'][:50]}...")
            print(f"   Prediction: {result['prediction']} ({result['spam_probability']:.2%})")
    
    else:
        # Interactive mode
        print("Email Spam Detector (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            email = input("\nEnter email text: ").strip()
            if email.lower() == 'quit':
                break
            
            if not email:
                continue
            
            result = detector.predict(email)
            print(f"\nPrediction: {result['prediction'].upper()}")
            print(f"Spam Probability: {result['spam_probability']:.2%}")
            print(f"Confidence: {'HIGH' if abs(result['spam_probability'] - 0.5) > 0.3 else 'MEDIUM' if abs(result['spam_probability'] - 0.5) > 0.1 else 'LOW'}")

if __name__ == "__main__":
    main()