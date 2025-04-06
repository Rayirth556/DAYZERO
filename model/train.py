import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocess import create_preprocessor

# Paths
DATA_FILE = "data.csv"
PREPROCESSOR_PATH = "model/saved/preprocessor.pkl"
MODEL_PATH = "model/saved/stock_predictor.pkl"

def train_model():
    df = pd.read_csv(DATA_FILE)

    # Drop rows missing any critical input or label
    df = df.dropna(subset=["age", "yoe", "bonus_percent", "investment_expert", "label"])

    if df.empty:
        print("[ERROR] No data left after dropping missing values. Please check your dataset.")
        return

    # Ignore if 'prediction' column is missing
    X_raw = df.drop(columns=["name", "label", "prediction"], errors='ignore')
    y = df["label"]

    # Preprocess and train
    preprocessor = create_preprocessor()
    X = preprocessor.fit_transform(X_raw)

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("[INFO] Model training complete.")

if __name__ == "__main__":
    train_model()
