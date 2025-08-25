from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
from config import *

class FeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.max_len = MAX_LEN
    
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer"""
        self.tfidf.fit(texts)
    
    def extract_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        return self.tfidf.transform(texts).toarray()
    
    def fit_tokenizer(self, texts):
        """Fit tokenizer for LSTM"""
        self.tokenizer.fit_on_texts(texts)
    
    def extract_sequence_features(self, texts):
        """Extract sequence features for LSTM"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding='post')
    
    def encode_labels(self, labels):
        """Encode labels"""
        return labels
    
    def get_vocab_size(self):
        """Get vocabulary size for embedding layer"""
        return min(len(self.tokenizer.word_index) + 1, 5000)
    
    def save_models(self, models_dir):
        """Save TF-IDF vectorizer and tokenizer"""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.tfidf, os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
        joblib.dump(self.tokenizer, os.path.join(models_dir, 'tokenizer.joblib'))
        print("Feature extractor models saved!")