from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.data_preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from config import *

app = Flask(__name__)

# Initialize components
preprocessor = TextPreprocessor()
feature_extractor = FeatureExtractor()
models = {}

def load_models():
    """Load trained models"""
    try:
        # Load feature extractor components
        feature_extractor.tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
        feature_extractor.tokenizer = joblib.load(os.path.join(MODELS_DIR, 'tokenizer.joblib'))
        
        # Load only Logistic Regression and LSTM models
        ml_models = ['logistic_regression']
        for name in ml_models:
            try:
                models[name] = joblib.load(os.path.join(MODELS_DIR, f'{name}_model.joblib'))
            except FileNotFoundError:
                print(f"Model {name} not found")
        
        # Load LSTM model
        try:
            models['lstm'] = load_model(os.path.join(MODELS_DIR, 'lstm_model.keras'))
        except FileNotFoundError:
            print("LSTM model not found")
            
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def predict_sentiment(text, model_name='logistic_regression'):
    """Predict sentiment for given text"""
    # Clean text
    cleaned_text = preprocessor.clean_text(text)
    
    if model_name in ['logistic_regression']:
        # TF-IDF features
        features = feature_extractor.extract_tfidf_features([cleaned_text])
        prediction = models[model_name].predict(features)[0]
        probabilities = models[model_name].predict_proba(features)[0]
    
    elif model_name == 'lstm':
        # Sequence features
        sequences = feature_extractor.extract_sequence_features([cleaned_text])
        probabilities = models[model_name].predict(sequences)[0]
        prediction = np.argmax(probabilities)
    
    else:
        return None, None
    
    # Map back to sentiment labels
    sentiment_labels = {
        0: 'Negative' if USE_3_CLASS else '0 - Negative',
        1: 'Neutral' if USE_3_CLASS else '2 - Neutral',
        2: 'Positive' if USE_3_CLASS else '4 - Positive'
    }
    
    if not USE_3_CLASS:
        sentiment_labels.update({
            1: '1 - Somewhat Negative',
            3: '3 - Somewhat Positive'
        })
    
    return sentiment_labels.get(prediction, 'Unknown'), probabilities.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_name = data.get('model', 'logistic_regression')
        
        if not text.strip():
            return jsonify({'error': 'Please enter some text'})
        
        sentiment, probabilities = predict_sentiment(text, model_name)
        
        if sentiment is None:
            return jsonify({'error': 'Model not found'})
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'probabilities': probabilities,
            'model': model_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/models')
def get_models():
    available_models = list(models.keys())
    return jsonify({'models': available_models})

@app.route('/config')
def get_config():
    """Return configuration settings"""
    return jsonify({
        'use_3_class': USE_3_CLASS
    })

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)