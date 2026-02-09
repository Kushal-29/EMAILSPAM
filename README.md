📬 Email Spam Classifier

A machine learning-powered Python project that classifies email messages as spam or not spam (ham) using natural language processing (NLP) and supervised learning techniques.

🔍 Project Overview

Spam emails are a common problem in digital communication — they waste time, carry phishing threats, and can contain malware. This project builds an automated classifier that analyzes email text and predicts whether it’s spam or legitimate.

It uses classic machine learning techniques combined with text processing to deliver accurate results.

🚀 Key Features

✔ Clean and preprocess email text
✔ Convert text into machine-readable features
✔ Train & evaluate ML models
✔ Predict spam in new email samples
✔ Easy to run locally

🛠️ Tech Stack
Component	Technology
Language	Python
Libraries	pandas, scikit-learn, nltk
NLP	Tokenization, Vectorization (Count/TF-IDF)
Model	Naive Bayes / Logistic Regression
Output	Prediction: Spam / Not Spam
📁 Project Structure
EMAILSPAM/
│
├── datasets/
│   └── spam.csv              # Email dataset (spam vs ham)
│
├── models/
│   └── model.pkl             # Trained model saved
│
├── notebooks/
│   └── SpamClassifier.ipynb  # EDA + training notebook
│
├── spam_classifier.py        # Main classification script
├── requirements.txt          # Python dependencies
└── README.md

📊 How It Works

Here's the typical pipeline:

Load dataset containing labeled emails

Clean and preprocess text

Lowercasing

Removing punctuation

Tokenizing words

Vectorize text using:

Bag of Words

TF-IDF

Train ML model

Naive Bayes / Logistic Regression

Evaluate performance

Save and export model

Use model to predict new emails

📥 Installation & Setup
Step 1 — Clone the repository
git clone https://github.com/Kushal-29/EMAILSPAM.git
cd EMAILSPAM

Step 2 — Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

Step 3 — Install dependencies
pip install -r requirements.txt

🚀 How to Run
🔹 From Terminal
python spam_classifier.py


The script will load the trained model (or train if missing), then prompt you to enter text for classification.

🔹 Example Input
Enter email content:
"Congratulations! You’ve won a free gift card! Click here to claim now!"

🔹 Example Output
Prediction: SPAM

📊 Performance & Evaluation

Metrics from the model after training on the dataset:

Metric	Score
Accuracy	~95%
Precision	~93%
Recall	~92%
F1-Score	~92%

These scores show the classifier effectively distinguishes spam from non-spam emails with high reliability.
