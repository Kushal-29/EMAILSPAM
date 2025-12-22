"""
Quick start script for Email Spam Detection
"""
import os
import subprocess
import sys

def run_command(command):
    """Run a command and print output"""
    print(f"\n💻 Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"⚠️  {result.stderr}")
    return result.returncode

def main():
    print("=" * 60)
    print("EMAIL SPAM DETECTOR - QUICK START")
    print("=" * 60)
    
    # Step 1: Check dataset
    print("\n📂 Checking dataset...")
    if not os.path.exists('data/spam.csv'):
        print("❌ Dataset not found at 'data/spam.csv'")
        print("\nPlease download the dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
        print("2. Download spam.csv")
        print("3. Place it in the 'data' folder")
        return
    
    print("✅ Dataset found!")
    
    # Step 2: Install requirements
    print("\n📦 Installing requirements...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    # Step 3: Download NLTK data
    print("\n📥 Downloading NLTK data...")
    run_command(f"{sys.executable} -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
    
    # Step 4: Create models directory
    print("\n📁 Creating directories...")
    os.makedirs('models', exist_ok=True)
    
    # Step 5: Train model
    print("\n🤖 Training model...")
    run_command(f"{sys.executable} train.py")
    
    # Step 6: Test prediction
    print("\n🧪 Testing prediction...")
    test_email = "Congratulations! You won a free iPhone. Click to claim!"
    run_command(f'{sys.executable} -c "from src.inference import SpamDetector; d = SpamDetector(); r = d.predict(\'{test_email}\'); print(f\"Test: {r[\"prediction\"]} (Spam prob: {r[\"spam_probability\"]:.2%})\")"')
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run web app: python app.py")
    print("2. Test predictions: python predict.py")
    print("3. Or use: python quick_start.py")

if __name__ == "__main__":
    main()