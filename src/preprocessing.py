import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def create_features_and_target(df: pd.DataFrame):
    df = df.copy()

    df["y_encoded"] = df["y"].map({"no": 0, "yes": 1})

    X = df.drop(columns=["y", "y_encoded", "duration"])
    y = df["y_encoded"]

    return X, y


def get_feature_columns():
    numeric_features = ["age", "balance", "day", "campaign", "pdays", "previous"]

    categorical_features = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]

    return numeric_features, categorical_features


def build_preprocessor():
    numeric_features, categorical_features = get_feature_columns()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False
                ),
                categorical_features,
            ),
        ]
    )

    return preprocessor


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )