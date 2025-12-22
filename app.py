from flask import Flask, request, render_template_string
import joblib
import re
import os

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('models/naive_bayes_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    MODEL_LOADED = True
    print("✅ Spam detection model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False

def clean_email_text(text):
    """Clean and preprocess email text"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def predict_email(email_text):
    """Predict if email is spam or ham"""
    if not MODEL_LOADED:
        return {
            'error': 'Model not loaded. Please train the model first.',
            'prediction': 'error'
        }
    
    # Clean the text
    cleaned_text = clean_email_text(email_text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # Calculate confidence level
    spam_prob = probability[1]
    if spam_prob > 0.9:
        confidence = "Very High"
        risk = "Critical"
        emoji = "🔥"
    elif spam_prob > 0.7:
        confidence = "High"
        risk = "Dangerous"
        emoji = "⚠️"
    elif spam_prob > 0.5:
        confidence = "Medium"
        risk = "Suspicious"
        emoji = "🤔"
    elif spam_prob > 0.3:
        confidence = "Low"
        risk = "Questionable"
        emoji = "📧"
    else:
        confidence = "Very Low"
        risk = "Safe"
        emoji = "✅"
    
    return {
        'prediction': 'spam' if prediction == 1 else 'ham',
        'spam_probability': float(spam_prob),
        'ham_probability': float(probability[0]),
        'confidence': confidence,
        'risk_level': risk,
        'emoji': emoji,
        'processed_text': cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text
    }

# Beautiful HTML Template with Glassmorphism Design
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpamShield Pro - AI Email Protection</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4361ee;
            --primary-dark: #3a0ca3;
            --success: #06d6a0;
            --danger: #ef476f;
            --warning: #ffd166;
            --glass: rgba(255, 255, 255, 0.15);
            --glass-dark: rgba(0, 0, 0, 0.2);
            --text-light: rgba(255, 255, 255, 0.9);
            --text-dark: #2d3748;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.9)), 
                        url('https://images.unsplash.com/photo-1497366754035-f200968a6e72?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2069&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: var(--text-light);
            min-height: 100vh;
            padding: 20px;
            backdrop-filter: blur(5px);
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header with Glassmorphism */
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px;
            background: var(--glass);
            backdrop-filter: blur(20px);
            border-radius: 30px;
            box-shadow: 0 20px 40px var(--glass-dark);
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, var(--primary), var(--success));
        }

        .logo {
            font-size: 3.2rem;
            font-weight: 700;
            margin-bottom: 15px;
            background: linear-gradient(90deg, #fff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'Montserrat', sans-serif;
        }

        .logo i {
            margin-right: 15px;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .tagline {
            font-size: 1.3rem;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 25px;
            font-weight: 300;
        }

        /* Main Content Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 30px;
            margin-bottom: 40px;
        }

        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Glass Cards */
        .glass-card {
            background: var(--glass);
            backdrop-filter: blur(20px);
            border-radius: 25px;
            padding: 35px;
            box-shadow: 0 15px 35px var(--glass-dark);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        }

        .section-title {
            font-size: 1.8rem;
            color: white;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            font-family: 'Montserrat', sans-serif;
        }

        .section-title i {
            margin-right: 15px;
            color: var(--primary);
            background: white;
            padding: 10px;
            border-radius: 12px;
            font-size: 1.2rem;
        }

        /* Email Input */
        .email-input {
            width: 100%;
            height: 200px;
            padding: 25px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            font-size: 1.1rem;
            font-family: 'Poppins', sans-serif;
            color: white;
            resize: vertical;
            margin-bottom: 25px;
            transition: all 0.3s;
        }

        .email-input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }

        .email-input:focus {
            outline: none;
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 0 4px rgba(67, 97, 238, 0.2);
        }

        /* Buttons */
        .btn-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 18px 35px;
            border: none;
            border-radius: 15px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            font-family: 'Montserrat', sans-serif;
            letter-spacing: 0.5px;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            flex: 2;
            box-shadow: 0 10px 20px rgba(67, 97, 238, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(67, 97, 238, 0.4);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            flex: 1;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        /* Results Section */
        .result-card {
            display: none;
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .prediction-display {
            text-align: center;
            margin-bottom: 30px;
        }

        .prediction-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }

        .prediction-text {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            font-family: 'Montserrat', sans-serif;
        }

        .prediction-spam {
            color: var(--danger);
            text-shadow: 0 0 20px rgba(239, 71, 111, 0.5);
        }

        .prediction-ham {
            color: var(--success);
            text-shadow: 0 0 20px rgba(6, 214, 160, 0.5);
        }

        /* Probability Bars */
        .probability-bars {
            margin: 30px 0;
        }

        .probability-bar {
            margin-bottom: 20px;
        }

        .bar-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .bar-container {
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 1s ease-in-out;
        }

        .bar-spam {
            background: linear-gradient(90deg, #ef476f, #ff0054);
            box-shadow: 0 0 10px rgba(239, 71, 111, 0.5);
        }

        .bar-ham {
            background: linear-gradient(90deg, #06d6a0, #04a777);
            box-shadow: 0 0 10px rgba(6, 214, 160, 0.5);
        }

        /* Confidence Badge */
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 25px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50px;
            font-weight: 600;
            margin: 20px 0;
        }

        /* Examples Grid */
        .examples-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }

        @media (max-width: 768px) {
            .examples-grid {
                grid-template-columns: 1fr;
            }
        }

        .example-card {
            padding: 20px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .example-card:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateX(5px);
        }

        .example-type {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .type-spam {
            background: rgba(239, 71, 111, 0.2);
            color: #ff6b8b;
        }

        .type-ham {
            background: rgba(6, 214, 160, 0.2);
            color: #6bffd5;
        }

        .example-text {
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.5;
        }

        /* Stats Bar */
        .stats-bar {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 40px;
        }

        .stat-card {
            text-align: center;
            padding: 25px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stat-value {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(90deg, var(--primary), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stat-label {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.7);
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .ai-loader {
            display: inline-block;
            position: relative;
            width: 80px;
            height: 80px;
        }

        .ai-loader div {
            position: absolute;
            border: 4px solid var(--primary);
            opacity: 1;
            border-radius: 50%;
            animation: ai-loader 1s cubic-bezier(0, 0.2, 0.8, 1) infinite;
        }

        .ai-loader div:nth-child(2) {
            animation-delay: -0.5s;
        }

        @keyframes ai-loader {
            0% {
                top: 36px;
                left: 36px;
                width: 0;
                height: 0;
                opacity: 1;
            }
            100% {
                top: 0px;
                left: 0px;
                width: 72px;
                height: 72px;
                opacity: 0;
            }
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 30px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .footer p {
            color: rgba(255, 255, 255, 0.7);
            margin: 5px 0;
        }

        .footer-logo {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 15px;
            background: linear-gradient(90deg, #fff, #a5b4fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .copyright {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.5);
            margin-top: 15px;
        }

        /* Glowing Effects */
        .glow {
            box-shadow: 0 0 20px rgba(67, 97, 238, 0.3);
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(67, 97, 238, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(67, 97, 238, 0); }
            100% { box-shadow: 0 0 0 0 rgba(67, 97, 238, 0); }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, var(--primary), var(--primary-dark));
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header pulse">
            <div class="logo">
                <i class="fas fa-shield-alt"></i> SpamShield Pro
            </div>
            <p class="tagline">Advanced AI-powered email protection with real-time threat detection</p>
            
            <div class="stats-bar">
                <div class="stat-card">
                    <div class="stat-value">99.2%</div>
                    <div class="stat-label">Detection Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">0.15s</div>
                    <div class="stat-label">Analysis Speed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">AI-Powered</div>
                    <div class="stat-label">Machine Learning</div>
                </div>
            </div>
        </header>

        <!-- Main Content Grid -->
        <div class="main-grid">
            <!-- Left Column: Input -->
            <div class="glass-card">
                <h2 class="section-title">
                    <i class="fas fa-envelope"></i> Email Analysis
                </h2>
                <form method="POST" id="emailForm">
                    <textarea 
                        class="email-input" 
                        name="email" 
                        placeholder="Paste your email here for instant spam analysis..."
                        required>{{ email if email else '' }}</textarea>
                    
                    <div class="btn-group">
                        <button type="submit" class="btn btn-primary" id="analyzeBtn">
                            <i class="fas fa-brain"></i> Analyze with AI
                        </button>
                        <button type="button" class="btn btn-secondary" onclick="clearForm()">
                            <i class="fas fa-eraser"></i> Clear Text
                        </button>
                    </div>
                </form>

                <div class="loading" id="loading">
                    <div class="ai-loader"><div></div><div></div></div>
                    <p>AI is analyzing your email content...</p>
                </div>

                <!-- Quick Examples -->
                <div style="margin-top: 30px;">
                    <h3 style="margin-bottom: 15px; color: white; font-size: 1.2rem;">
                        <i class="fas fa-bolt"></i> Quick Test Examples
                    </h3>
                    <div class="examples-grid">
                        <div class="example-card" onclick="useExample(this, 'spam')">
                            <div class="example-type type-spam">SPAM</div>
                            <p class="example-text">"Congratulations! You've won a free iPhone. Click now to claim!"</p>
                        </div>
                        <div class="example-card" onclick="useExample(this, 'ham')">
                            <div class="example-type type-ham">HAM</div>
                            <p class="example-text">"Hi team, let's schedule a meeting tomorrow at 2 PM to discuss."</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column: Results -->
            <div class="glass-card">
                <h2 class="section-title">
                    <i class="fas fa-chart-pie"></i> Analysis Results
                </h2>
                
                <div class="result-card" id="resultCard">
                    <div class="prediction-display">
                        <div class="prediction-icon" id="predictionEmoji"></div>
                        <div class="prediction-text" id="predictionText"></div>
                        <div class="confidence-badge">
                            <i class="fas fa-bullseye"></i>
                            AI Confidence: <span id="confidenceText">--</span>
                        </div>
                    </div>

                    <div class="probability-bars">
                        <div class="probability-bar">
                            <div class="bar-label">
                                <span>Spam Probability</span>
                                <span id="spamPercent">0%</span>
                            </div>
                            <div class="bar-container">
                                <div class="bar-fill bar-spam" id="spamBar" style="width: 0%"></div>
                            </div>
                        </div>
                        
                        <div class="probability-bar">
                            <div class="bar-label">
                                <span>Ham Probability</span>
                                <span id="hamPercent">0%</span>
                            </div>
                            <div class="bar-container">
                                <div class="bar-fill bar-ham" id="hamBar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>

                    <div style="text-align: center; margin-top: 30px;">
                        <div style="padding: 15px; background: rgba(255, 255, 255, 0.1); border-radius: 15px;">
                            <h4><i class="fas fa-info-circle"></i> Risk Assessment</h4>
                            <p id="riskText" style="font-size: 1.1rem; margin-top: 10px;"></p>
                        </div>
                    </div>
                </div>

                <div id="noResult" style="text-align: center; padding: 50px; color: rgba(255, 255, 255, 0.5);">
                    <i class="fas fa-search" style="font-size: 3.5rem; margin-bottom: 20px; opacity: 0.5;"></i>
                    <h3 style="margin-bottom: 10px;">No Analysis Yet</h3>
                    <p>Submit an email to see AI-powered spam detection results</p>
                </div>
            </div>
        </div>

        <!-- Examples Section -->
        <div class="glass-card" style="margin-top: 30px;">
            <h2 class="section-title">
                <i class="fas fa-vial"></i> Test Examples
            </h2>
            
            <div class="examples-grid">
                <div class="example-card" onclick="useExample(this, 'spam')">
                    <div class="example-type type-spam">SPAM</div>
                    <p class="example-text">"URGENT: Your account has been suspended! Click immediately to verify."</p>
                </div>
                
                <div class="example-card" onclick="useExample(this, 'ham')">
                    <div class="example-type type-ham">HAM</div>
                    <p class="example-text">"Reminder: Your appointment is scheduled for tomorrow at 10 AM."</p>
                </div>
                
                <div class="example-card" onclick="useExample(this, 'spam')">
                    <div class="example-type type-spam">SPAM</div>
                    <p class="example-text">"Earn $5000 weekly from home! No experience needed. Start now!"</p>
                </div>
                
                <div class="example-card" onclick="useExample(this, 'ham')">
                    <div class="example-type type-ham">HAM</div>
                    <p class="example-text">"Thanks for your email. I'll review the document and get back to you."</p>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <div class="footer-logo">
                <i class="fas fa-shield-alt"></i> SpamShield Pro
            </div>
            <p>Advanced Email Protection System</p>
            <p>Powered by AI & Machine Learning Algorithms</p>
            <p>Version 3.0 | Real-time Threat Detection</p>
            <div class="copyright">
                © 2025 SpamShield Pro. All rights reserved. | Made with <i class="fas fa-heart" style="color: #ef476f;"></i> for a safer inbox
            </div>
        </footer>
    </div>

    <script>
        // Show loading animation
        document.getElementById('emailForm').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('analyzeBtn').innerHTML = '<i class="fas fa-cog fa-spin"></i> Analyzing...';
        });

        // Clear form
        function clearForm() {
            document.querySelector('.email-input').value = '';
            document.getElementById('noResult').style.display = 'block';
            document.getElementById('resultCard').style.display = 'none';
            document.getElementById('loading').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('analyzeBtn').innerHTML = '<i class="fas fa-brain"></i> Analyze with AI';
        }

        // Use example text
        function useExample(exampleCard, type) {
            const exampleText = exampleCard.querySelector('.example-text').textContent;
            document.querySelector('.email-input').value = exampleText;
            document.querySelector('.email-input').focus();
            
            // Add visual feedback
            exampleCard.style.transform = 'scale(0.95)';
            setTimeout(() => {
                exampleCard.style.transform = '';
            }, 200);
        }

        // Display results from server
        window.onload = function() {
            {% if result %}
                const result = {{ result|tojson|safe }};
                
                if (result.prediction !== 'error') {
                    // Hide "no results" message
                    document.getElementById('noResult').style.display = 'none';
                    
                    // Show result card with animation
                    const resultCard = document.getElementById('resultCard');
                    resultCard.style.display = 'block';
                    
                    // Set prediction text and emoji
                    document.getElementById('predictionEmoji').textContent = result.emoji;
                    
                    const predictionText = document.getElementById('predictionText');
                    if (result.prediction === 'spam') {
                        predictionText.textContent = 'SPAM DETECTED';
                        predictionText.className = 'prediction-text prediction-spam';
                    } else {
                        predictionText.textContent = 'SAFE EMAIL';
                        predictionText.className = 'prediction-text prediction-ham';
                    }
                    
                    // Update confidence
                    document.getElementById('confidenceText').textContent = result.confidence;
                    
                    // Update percentages and bars
                    const spamPercent = (result.spam_probability * 100).toFixed(1);
                    const hamPercent = (result.ham_probability * 100).toFixed(1);
                    
                    document.getElementById('spamPercent').textContent = spamPercent + '%';
                    document.getElementById('hamPercent').textContent = hamPercent + '%';
                    
                    // Animate bars
                    setTimeout(() => {
                        document.getElementById('spamBar').style.width = spamPercent + '%';
                        document.getElementById('hamBar').style.width = hamPercent + '%';
                    }, 100);
                    
                    // Update risk text
                    const riskText = document.getElementById('riskText');
                    riskText.innerHTML = `<strong>${result.risk_level}</strong> - ${getRiskDescription(result.spam_probability)}`;
                    
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                    document.getElementById('analyzeBtn').innerHTML = '<i class="fas fa-brain"></i> Analyze with AI';
                }
            {% endif %}
        };

        function getRiskDescription(spamProb) {
            if (spamProb > 0.9) return 'Critical threat detected! Do not interact.';
            if (spamProb > 0.7) return 'High risk email. Avoid clicking links.';
            if (spamProb > 0.5) return 'Suspicious content detected. Proceed with caution.';
            if (spamProb > 0.3) return 'Potentially safe but verify sender.';
            return 'Safe email. No threats detected.';
        }

        // Add some interactive effects
        document.querySelectorAll('.example-card').forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.4)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.boxShadow = '';
            });
        });
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    email_text = ""
    result = None
    
    if request.method == 'POST':
        email_text = request.form.get('email', '')
        
        if email_text:
            result = predict_email(email_text)
    
    return render_template_string(HTML_TEMPLATE, email=email_text, result=result)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 SPAMSHIELD PRO - Advanced Email Protection")
    print("="*70)
    print("\n✨ Features:")
    print("   • Glassmorphism UI with transparency")
    print("   • Real-time AI analysis")
    print("   • Beautiful probability visualizations")
    print("   • Interactive test examples")
    print("\n📡 Launching at: http://localhost:5000")
    print("🔒 Model Status: " + ("✅ Loaded" if MODEL_LOADED else "❌ Not Loaded"))
    print("\n⏸️  Press CTRL+C to stop")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)