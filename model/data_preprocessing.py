import pandas as pd
from preprocess import fit_and_save_preprocessor
import joblib

DATA_FILE = "data.csv"
PREPROCESSOR_PATH = "model/saved/preprocessor.pkl"

def prepare_data():
    df = pd.read_csv(DATA_FILE)
    fit_and_save_preprocessor(df, PREPROCESSOR_PATH)

    # Optional: Return transformed data if needed
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_transformed = preprocessor.transform(df.drop(columns=["name"]))
    return X_transformed

if __name__ == "__main__":
    prepare_data()
