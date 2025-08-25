# Movie Sentiment Analysis

A Flask-based web application for 3-class sentiment analysis (Negative, Neutral, Positive) of movie reviews using the Stanford Sentiment Treebank (SST) dataset. The app uses Logistic Regression (~75-80% accuracy) and LSTM (~80-85% accuracy) models, trained on a subsampled dataset (~10,000 samples, 4,000 Neutral, 3,000 Negative/Positive) in ~10 minutes.

Developed during Ezitech Internship, August 2025. Powered by xAI.

## Directory Structure
```
G:\Ezitech Internship\moviesent-project\
├── data\
│   ├── train.tsv
│   ├── test.tsv
│   ├── glove.6B.100d.txt
├── models\
│   ├── tfidf_vectorizer.joblib
│   ├── tokenizer.joblib
│   ├── logistic_regression_model.joblib
│   ├── lstm_model.keras
├── src\
|    ├── templates\
|      ├── index.html
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── app.py
├── config.py
├── main.py
├── README.md
```

## Requirements
- Python 3.8+
- Dependencies:
  ```bash
  pip install pandas numpy scikit-learn tensorflow nltk joblib flask
  ```
- NLTK data:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```
- Dataset: `train.tsv`, `test.tsv` (SST dataset), `glove.6B.100d.txt` (GloVe embeddings) in `data/`.

## Setup
1. **Clone or Set Up Directory**:
   - Ensure the directory structure matches the above.
   - Place `train.tsv`, `test.tsv`, and `glove.6B.100d.txt` in `data/`.

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn tensorflow nltk joblib flask
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

3. **Fix Protobuf (if warnings occur)**:
   ```bash
   pip install protobuf==5.28.3
   ```

## Usage
1. **Train Models**:
   - Run:
     ```bash
     python main.py --train
     ```
   - Generates `logistic_regression_model.joblib`, `lstm_model.keras`, `tfidf_vectorizer.joblib`, `tokenizer.joblib` in `models/`.
   - Time: ~3-5 minutes (retraining) or ~6-9 minutes (from scratch).
   - Expected accuracy: Logistic Regression ~75-80%, LSTM ~80-85%.

2. **Evaluate Models**:
   - Run:
     ```bash
     python main.py --evaluate
     ```
   - Outputs accuracy, confusion matrix, and classification report for Neutral class.

3. **Run Web App**:
   - Run:
     ```bash
     python main.py --run-app
     ```
   - Open `http://127.0.0.1:5000` in a browser.
   - Enter a review, select a model (`logistic_regression` or `lstm`), and click "Analyze Sentiment".

## Testing
Test the app with these reviews:
- **Negative**: "The movie was a complete disaster, with terrible dialogue and no coherent story." (Expect: Negative, ~85% Negative probability)
- **Neutral**: "It was an average film, nothing special but watchable enough." (Expect: Neutral, ~50-60% Neutral probability)
- **Positive**: "A fantastic movie with brilliant performances and an inspiring plot!" (Expect: Positive, ~85% Positive probability)


## Notes
- SOTA for SST-3 is ~80-85% with BERT, so 80-85% is realistic. <grok:render type="render_inline_citation"><argument name="citation_id">21</argument></grok:render>
- Contact: Developed during Internship, August 2025.