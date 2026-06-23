from src.preprocessing import load_data, create_features_and_target, split_data
from src.train import train_model
from src.evaluate import evaluate_model


DATA_PATH = "data/bank-full.csv"


def main():
    df = load_data(DATA_PATH)

    X, y = create_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()