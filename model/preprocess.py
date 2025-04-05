from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    # Identify feature types
    numeric_features = ["age", "salary", "yoe", "bonus_percent"]
    categorical_features = ["position", "investment_expert"]

    # Pipelines for numeric and categorical
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, preprocessor
