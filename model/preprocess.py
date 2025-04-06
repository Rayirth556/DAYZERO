from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

def create_preprocessor():
    numeric_features = ["age", "salary", "yoe", "bonus_percent"]
    categorical_features = ["position", "investment_expert"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def fit_and_save_preprocessor(df: pd.DataFrame, save_path: str):
    df_features = df.drop(columns=["name"])
    preprocessor = create_preprocessor()
    preprocessor.fit(df_features)
    joblib.dump(preprocessor, save_path)
    print(f"[INFO] Preprocessor saved to: {save_path}")
