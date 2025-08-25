# data_preprocessing.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
import os

# Add parent directory to path to import config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import *

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    def __init__(self):
        # Use minimal stopwords - keep words that might carry sentiment
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that'
        }
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Less aggressive text cleaning - preserve sentiment cues"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Keep basic punctuation that carries sentiment (! ? , . ')
        # Remove only special characters, not all punctuation
        text = re.sub(r'[^a-zA-Z\s\!\?\,\.\']', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove minimal stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 1
        ]
        
        return ' '.join(tokens)
    
    def map_sentiment(self, sentiment):
        """Map 5-point scale to 3 classes if enabled"""
        if USE_3_CLASS:
            if sentiment in [0, 1]:
                return 0  # Negative
            elif sentiment == 2:
                return 1  # Neutral
            else:
                return 2  # Positive
        return sentiment

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("Loading data...")
    
    # Load data
    train_df = pd.read_csv(TRAIN_DATA_PATH, sep=CSV_SEPARATOR)
    test_df = pd.read_csv(TEST_DATA_PATH, sep=CSV_SEPARATOR)
    
    print(f"Original training data shape: {train_df.shape}")
    print(f"Original test data shape: {test_df.shape}")
    print(f"Training columns: {train_df.columns.tolist()}")
    print(f"Test columns: {test_df.columns.tolist()}")
    
    # Check if the test data has sentiment labels
    has_test_sentiment = 'Sentiment' in test_df.columns
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Clean text with less aggressive approach
    print("Cleaning text data (less aggressive)...")
    train_df['cleaned_phrase'] = train_df['Phrase'].apply(preprocessor.clean_text)
    test_df['cleaned_phrase'] = test_df['Phrase'].apply(preprocessor.clean_text)
    
    # Map sentiment for training data
    if USE_3_CLASS:
        train_df['sentiment'] = train_df['Sentiment'].apply(preprocessor.map_sentiment)
        num_classes = 3
    else:
        train_df['sentiment'] = train_df['Sentiment']
        num_classes = 5
    
    # Map sentiment for test data only if it has sentiment labels
    if has_test_sentiment:
        if USE_3_CLASS:
            test_df['sentiment'] = test_df['Sentiment'].apply(preprocessor.map_sentiment)
        else:
            test_df['sentiment'] = test_df['Sentiment']
        print("Test data contains sentiment labels")
    else:
        # Test data doesn't have sentiment labels (this is normal for competition datasets)
        test_df['sentiment'] = None
        print("Test data does not contain sentiment labels (expected for prediction)")
    
    # Remove empty texts after cleaning, duplicates, and missing
    train_df = train_df[train_df['cleaned_phrase'].str.len() > 0]
    train_df.drop_duplicates(subset=['cleaned_phrase'], inplace=True)
    train_df.dropna(subset=['cleaned_phrase', 'sentiment'], inplace=True)
    
    test_df = test_df[test_df['cleaned_phrase'].str.len() > 0]
    
    print(f"Processed training data shape: {train_df.shape}")
    print(f"Processed test data shape: {test_df.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Show sentiment distribution
    print("\nTraining sentiment distribution:")
    print(train_df['Sentiment'].value_counts().sort_index())
    if USE_3_CLASS:
        print("\nTraining sentiment distribution (3-class mapped):")
        print(train_df['sentiment'].value_counts().sort_index())
    
    # Debug: Show sample cleaning results
    print("\n=== SAMPLE CLEANING RESULTS ===")
    sample_indices = [0, 1, 2, 3, 4]  # First 5 samples
    for idx in sample_indices:
        original = train_df['Phrase'].iloc[idx]
        cleaned = train_df['cleaned_phrase'].iloc[idx]
        sentiment = train_df['Sentiment'].iloc[idx]
        print(f"Original: '{original}'")
        print(f"Cleaned:  '{cleaned}'")
        print(f"Sentiment: {sentiment}")
        print("-" * 50)
    
    if has_test_sentiment:
        print("\nTest sentiment distribution:")
        print(test_df['Sentiment'].value_counts().sort_index())
        if USE_3_CLASS:
            print("\nTest sentiment distribution (3-class mapped):")
            print(test_df['sentiment'].value_counts().sort_index())
    
    return train_df, test_df, num_classes, has_test_sentiment

if __name__ == "__main__":
    train_df, test_df, num_classes, has_test_sentiment = load_and_preprocess_data()
    print("\nSample cleaned training data:")
    print(train_df[['Phrase', 'cleaned_phrase', 'Sentiment', 'sentiment']].head(10))
    
    print("\nSample test data:")
    if has_test_sentiment:
        print(test_df[['Phrase', 'cleaned_phrase', 'Sentiment', 'sentiment']].head(5))
    else:
        print(test_df[['Phrase', 'cleaned_phrase']].head(5))

