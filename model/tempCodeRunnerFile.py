from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import os

def train_model():
    df = pd.read_csv("data.csv")

    # Only train on known labels
    df = df[df["label"].isin(["increase", "decrease"])]

    X = df.drop(columns=["label", "name"])
    y = df["label"]

    categorical_features = ["position", "investment_expert"]
    numerical_features = ["salary", "yoe", "bonus_percent"]

    # Explicit categories for OneHotEncoder
    encoder = OneHotEncoder(
        categories=[
            ["Intern", "Junior", "Senior", "Manager"],     # position
            ["yes", "no"]                                  # investment_expert
        ],
        handle_unknown='ignore'
    )

    preprocessor = ColumnTransformer([
        ("cat", encoder, categorical_features),
        ("num", StandardScaler(), numerical_features),
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)

    with open("report.txt", "w") as f:
        f.write(report)

    os.makedirs("model/saved", exist_ok=True)
    joblib.dump(model_pipeline, "model/saved/model.pkl")

if __name__ == "__main__":
    train_model()
