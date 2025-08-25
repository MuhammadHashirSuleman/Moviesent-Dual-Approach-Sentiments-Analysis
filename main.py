from src.app import app
from src.model_training import main as train_models
from src.evaluation import main as evaluate_models
import argparse

def main():
    parser = argparse.ArgumentParser(description='Movie Sentiment Analysis Project')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--run-app', action='store_true', help='Run Flask web app')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training models...")
        train_models()
    
    if args.evaluate:
        print("Evaluating models...")
        evaluate_models()
    
    if args.run_app:
        print("Starting web application...")
        from src.app import load_models
        load_models()
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    if not any([args.train, args.evaluate, args.run_app]):
        print("Please specify an action: --train, --evaluate, or --run-app")

if __name__ == "__main__":
    main()