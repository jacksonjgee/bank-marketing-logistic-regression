from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.preprocessing import build_preprocessor


def build_model():
    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    class_weight="balanced",
                    C=0.1
                )
            ),
        ]
    )

    return model


def train_model(X_train, y_train):
    model = build_model()
    model.fit(X_train, y_train)
    return model