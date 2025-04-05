import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from preprocess import preprocess_data

def train_model():
    # ✅ Load data
    df = pd.read_csv("data.csv")

    # ✅ Drop rows with missing target labels (most important fix)
    df = df.dropna(subset=["label"])

    # ✅ Optional: drop rows with any missing values
    # df = df.dropna()

    # ✅ Preprocess
    X, y, preprocessor = preprocess_data(df)

    # ✅ Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # ✅ Save model & preprocessor
    joblib.dump(preprocessor, "model/saved/preprocessor.pkl")
    joblib.dump(model, "model/saved/stock_predictor.pkl")

    print("✅ Model and preprocessor saved successfully.")

# Optional utility function
def load_and_clean_data(filepath="data.csv"):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["label"])  # ensure no NaNs in label
    return df

if __name__ == "__main__":
    train_model()
