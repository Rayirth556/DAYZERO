import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_preprocessing import load_and_clean_data, preprocess_data
import os

def train_model():
    df = load_and_clean_data("/home/godkiller/killerjack/data.csv")
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model and preprocessor
    os.makedirs("model/saved", exist_ok=True)
    joblib.dump(model, "model/saved/stock_predictor.pkl")
    joblib.dump(preprocessor, "model/saved/preprocessor.pkl")

if __name__ == "__main__":
    train_model()
