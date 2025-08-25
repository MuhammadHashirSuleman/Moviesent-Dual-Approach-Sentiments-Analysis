# model_training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample  # For balancing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional  # Added Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
from config import *
from .data_preprocessing import load_and_preprocess_data
from .feature_extraction import FeatureExtractor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.models = {}
    
    def train_ml_models(self, X_train, y_train, X_val, y_val):
        """Train traditional machine learning models"""
        print("Training ML models...")
        
        # TF-IDF features with n-grams
        self.feature_extractor.tfidf.set_params(ngram_range=(1, 3))  # Add n-grams as per requirements
        X_train_tfidf = self.feature_extractor.extract_tfidf_features(X_train)
        X_val_tfidf = self.feature_extractor.extract_tfidf_features(X_val)
        
        # Models with class weights (backup to balancing)
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            # Dropped RF and SVM as project requires only LR, but keep if wanted
        }
        
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_val_tfidf)
            accuracy = accuracy_score(y_val, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': classification_report(y_val, y_pred, output_dict=True)
            }
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        self.models.update(results)
        return results
    
    def load_glove_embeddings(self):
        """Load GloVe embeddings"""
        print("Loading GloVe embeddings...")
        embeddings_index = {}
        with open(GLOVE_PATH, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index
    
    def create_embedding_matrix(self, embeddings_index, vocab_size):
        """Create embedding matrix from GloVe"""
        word_index = self.feature_extractor.tokenizer.word_index
        embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i < vocab_size:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    def build_lstm_model(self, vocab_size, num_classes, embedding_matrix):
        """Build optimized LSTM model with GloVe"""
        model = Sequential([
            Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, input_length=MAX_LEN),
            Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
            Bidirectional(LSTM(64, dropout=0.3)),
            Dense(128, activation='relu'),  # Increased dense layers
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Lower LR for stability
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, num_classes):
        """Train LSTM model with class weights and GloVe"""
        print("Training LSTM model...")
        
        # Convert text to sequences
        X_train_seq = self.feature_extractor.extract_sequence_features(X_train)
        X_val_seq = self.feature_extractor.extract_sequence_features(X_val)
        
        # Encode labels
        y_train_enc = self.feature_extractor.encode_labels(y_train)
        y_val_enc = self.feature_extractor.encode_labels(y_val)
        
        # Load GloVe and create matrix
        embeddings_index = self.load_glove_embeddings()
        vocab_size = self.feature_extractor.get_vocab_size()
        embedding_matrix = self.create_embedding_matrix(embeddings_index, vocab_size)
        
        # Build model
        model = self.build_lstm_model(vocab_size, num_classes, embedding_matrix)
        
        # Calculate class weights if enabled
        class_weight = None
        if USE_CLASS_WEIGHTS:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_enc),
                y=y_train_enc
            )
            class_weight = dict(enumerate(class_weights))
            print(f"Class weights: {class_weight}")
        
        # Callbacks
        early_stopping = EarlyStopping(
            patience=7,  # Increased patience
            restore_best_weights=True,
            monitor='val_accuracy'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-5  # Lower min LR
        )
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_enc,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val_seq, y_val_enc),
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight,
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(X_val_seq, y_val_enc, verbose=0)
        print(f"LSTM Validation Accuracy: {accuracy:.4f}")
        
        self.models['lstm'] = {
            'model': model,
            'accuracy': accuracy,
            'history': history.history
        }
        
        return model, history
    
    def save_models(self):
        """Save all trained models"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        for name, model_info in self.models.items():
            if name != 'lstm':  # Save sklearn models
                joblib.dump(model_info['model'], f"{MODELS_DIR}/{name}_model.joblib")
            else:  # Save Keras model
                model_info['model'].save(f"{MODELS_DIR}/lstm_model.keras")
        
        # Save feature extractor
        self.feature_extractor.save_models(MODELS_DIR)
        print("Models saved successfully!")

def main():
    # Load and preprocess data
    train_df, test_df, num_classes, has_test_sentiment = load_and_preprocess_data()
    
    # Balance the data for 3 classes
    print("\nBalancing dataset...")
    df_neg = train_df[train_df['sentiment'] == 0]
    df_neu = train_df[train_df['sentiment'] == 1]
    df_pos = train_df[train_df['sentiment'] == 2]
    
    max_size = max(len(df_neg), len(df_neu), len(df_pos))  # Balance to largest (neutral)
    df_neg_upsampled = resample(df_neg, replace=True, n_samples=max_size, random_state=42)
    df_pos_upsampled = resample(df_pos, replace=True, n_samples=max_size, random_state=42)
    if len(df_neu) < max_size:  # If neutral is not max, upsample it too
        df_neu_upsampled = resample(df_neu, replace=True, n_samples=max_size, random_state=42)
    else:
        df_neu_upsampled = df_neu
    
    train_balanced = pd.concat([df_neg_upsampled, df_neu_upsampled, df_pos_upsampled])
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Balanced sentiment distribution:")
    print(train_balanced['sentiment'].value_counts())
    
    # Split training data for validation (70/30 as per project)
    X_train, X_val, y_train, y_val = train_test_split(
        train_balanced['cleaned_phrase'], train_balanced['sentiment'],
        test_size=0.3, random_state=42, stratify=train_balanced['sentiment']
    )
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train ML models (only LR as required)
    print("\n" + "="*50)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*50)
    ml_results = trainer.train_ml_models(X_train, y_train, X_val, y_val)
    
    # Train LSTM model
    print("\n" + "="*50)
    print("TRAINING LSTM MODEL")
    print("="*50)
    lstm_model, history = trainer.train_lstm(X_train, y_train, X_val, y_val, num_classes)
    
    # Save models
    trainer.save_models()
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for name, result in trainer.models.items():
        print(f"{name}: {result['accuracy']:.4f}")
    
    return trainer.models

if __name__ == "__main__":
    results = main()

