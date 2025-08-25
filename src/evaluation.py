import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_preprocess_data
from src.feature_extraction import FeatureExtractor
from config import *

def main():
    print("Loading and preprocessing training data for evaluation...")
    train_df, test_df, num_classes, has_test_sentiment = load_and_preprocess_data()
    
    print("\nBalancing dataset (same as training)...")
    df_neg = train_df[train_df['sentiment'] == 0]
    df_neu = train_df[train_df['sentiment'] == 1]
    df_pos = train_df[train_df['sentiment'] == 2]
    max_size = min(len(df_neg), len(df_neu), len(df_pos), 3000)  # Match training subsampling
    df_neg_upsampled = resample(df_neg, replace=True, n_samples=max_size, random_state=42)
    df_neu_upsampled = resample(df_neu, replace=True, n_samples=max_size, random_state=42)
    df_pos_upsampled = resample(df_pos, replace=True, n_samples=max_size, random_state=42)
    train_balanced = pd.concat([df_neg_upsampled, df_neu_upsampled, df_pos_upsampled])
    
    train_balanced = train_balanced[train_balanced['cleaned_phrase'].str.split().str.len() >= 5]  # Match training filter
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Balanced sentiment distribution:")
    print(train_balanced['sentiment'].value_counts())
    
    # Split into training and validation sets (same as model_training.py)
    X_train, X_val, y_train, y_val = train_test_split(
        train_balanced['cleaned_phrase'], train_balanced['sentiment'],
        test_size=0.3, random_state=42, stratify=train_balanced['sentiment']
    )
    
    print("Loading models...")
    extractor = FeatureExtractor()
    try:
        extractor.tfidf = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
        extractor.tokenizer = joblib.load(os.path.join(MODELS_DIR, 'tokenizer.joblib'))
        lr_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression_model.joblib'))
        lstm_model = load_model(os.path.join(MODELS_DIR, 'lstm_model.keras'))
    except FileNotFoundError as e:
        print(f"Error: Model file not found: {e}. Please run 'python main.py --train' first.")
        return
    
    print("Evaluating Logistic Regression...")
    X_val_tfidf = extractor.extract_tfidf_features(X_val)
    y_pred_lr = lr_model.predict(X_val_tfidf)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_val, y_pred_lr):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lr))
    print("Classification Report:\n", classification_report(y_val, y_pred_lr, target_names=['Negative', 'Neutral', 'Positive']))
    
    print("Evaluating LSTM...")
    X_val_seq = extractor.extract_sequence_features(X_val)
    y_pred_lstm = np.argmax(lstm_model.predict(X_val_seq), axis=1)
    print(f"LSTM Accuracy: {accuracy_score(y_val, y_pred_lstm):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_lstm))
    print("Classification Report:\n", classification_report(y_val, y_pred_lstm, target_names=['Negative', 'Neutral', 'Positive']))

if __name__ == "__main__":
    main()