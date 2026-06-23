# Bank Marketing Logistic Regression

A machine learning classification project using logistic regression to predict whether a bank customer will subscribe to a term deposit.

## Summary

This project uses the UCI Bank Marketing dataset to build a binary classification model for predicting term deposit subscription.

The final model uses a scikit-learn pipeline with preprocessing and logistic regression. Because the dataset is imbalanced, the model was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix rather than accuracy alone.

## Results

The balanced logistic regression model achieved the following results on the test set:

| Metric                         | Result |
| ------------------------------ | -----: |
| Accuracy                       |  75.4% |
| Precision for subscribed class |    27% |
| Recall for subscribed class    |    63% |
| F1-score for subscribed class  |    38% |

### Confusion Matrix

|            | Predicted No | Predicted Yes |
| ---------- | -----------: | ------------: |
| Actual No  |         6151 |          1834 |
| Actual Yes |          388 |           670 |

The model identified approximately **63% of actual subscribers**, which is useful in a marketing context where finding potential subscribers is important.

However, the precision for the subscribed class was approximately **27%**, meaning many predicted subscribers were actually non-subscribers. This shows a trade-off between recall and precision.

## Research Question

> Can customer characteristics and previous marketing campaign information be used to predict whether a customer subscribes to a term deposit?

## Dataset

The dataset was sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).

It contains **45,211 instances**, **16 input features**, and **1 binary target variable**.

### Target Variable

| Variable | Description                                     |
| -------- | ----------------------------------------------- |
| `y`      | Whether the client subscribed to a term deposit |

The target variable is binary:

* `yes` = subscribed
* `no` = did not subscribe

The dataset is imbalanced, with most clients not subscribing.

## Features

| Variable    | Type        | Description                                         |
| ----------- | ----------- | --------------------------------------------------- |
| `age`       | Numerical   | Client's age                                        |
| `job`       | Categorical | Client's occupation                                 |
| `marital`   | Categorical | Marital status                                      |
| `education` | Categorical | Education level                                     |
| `default`   | Binary      | Whether the client has credit in default            |
| `balance`   | Numerical   | Average yearly account balance                      |
| `housing`   | Binary      | Whether the client has a housing loan               |
| `loan`      | Binary      | Whether the client has a personal loan              |
| `contact`   | Categorical | Contact communication method                        |
| `day`       | Numerical   | Day of the month of the last contact                |
| `month`     | Categorical | Month of the last contact                           |
| `duration`  | Numerical   | Duration of the last contact in seconds             |
| `campaign`  | Numerical   | Number of contacts made during the current campaign |
| `pdays`     | Numerical   | Days since the client was previously contacted      |
| `previous`  | Numerical   | Number of contacts made before the current campaign |
| `poutcome`  | Categorical | Outcome of the previous marketing campaign          |

## Model

The model used in this project is **Logistic Regression** from scikit-learn.

A scikit-learn pipeline was used to combine preprocessing and model training into one workflow.

The model uses:

* `LogisticRegression`
* `class_weight="balanced"`
* `solver="liblinear"`
* `C=0.1`
* `max_iter=5000`

The balanced class weighting was used because the dataset contains many more non-subscribers than subscribers.

## Preprocessing

The preprocessing pipeline includes:

* Target encoding:

  * `no = 0`
  * `yes = 1`
* Removal of the `duration` column to avoid data leakage
* Scaling numerical features using `StandardScaler`
* One-hot encoding categorical features using `OneHotEncoder`
* Stratified train-test split to preserve the target class distribution

### Data Leakage Note

The `duration` feature was removed before modelling.

Although it is strongly related to the target variable, it is only known after the phone call has occurred. Including it would make the model less realistic for predicting whether a client is likely to subscribe before the call outcome is known.

## Exploratory Data Analysis

The exploratory data analysis is included in the notebook:

```text
notebooks/Exploratory Data Analysis.ipynb
```

### Key EDA Findings

* The target variable is imbalanced, with most clients not subscribing.
* Several numerical features, including `balance`, `campaign`, `pdays`, and `previous`, are right-skewed and contain potential outliers.
* The `duration` feature is strongly related to the target but was removed due to possible data leakage.
* Several categorical features showed meaningful differences in subscription rates, especially `poutcome`, `month`, `job`, `housing`, `loan`, and `contact`.
* The `"unknown"` category appears in several categorical features and was kept as its own category during encoding.

## Project Structure

```text
Bank_Marketing_Logistic_Regression/
├── data/
│   └── bank-full.csv
│
├── notebooks/
│   └── Exploratory Data Analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
├── requirements.txt
└── main.py
```

## File Responsibilities

| File                                        | Purpose                                                                                              |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `main.py`                                   | Runs the full machine learning workflow                                                              |
| `src/preprocessing.py`                      | Loads the data, prepares features and target, builds the preprocessing pipeline, and splits the data |
| `src/train.py`                              | Builds and trains the logistic regression model                                                      |
| `src/evaluate.py`                           | Evaluates the model using accuracy, confusion matrix, and classification report                      |
| `notebooks/Exploratory Data Analysis.ipynb` | Contains the full exploratory data analysis                                                          |

## Technologies Used

* Python
* pandas
* NumPy
* Matplotlib
* scikit-learn
* Jupyter Notebook

## How to Run

Clone the repository:

```bash
git clone https://github.com/jacksonjgee/Bank_Marketing_Logistic_Regression.git
cd Bank_Marketing_Logistic_Regression
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python3 main.py
```

## Future Improvements

Possible improvements include:

* Tuning the classification threshold to adjust the precision-recall trade-off
* Adding ROC-AUC and precision-recall curve analysis
* Testing other classification models such as decision trees, random forests, or gradient boosting
* Performing additional feature engineering
* Comparing model performance with and without `duration`
* Building an interactive Streamlit app for predictions

## Conclusion

This project demonstrates a complete beginner machine learning workflow, including data exploration, preprocessing, logistic regression modelling, and evaluation.

The final balanced logistic regression model was able to identify a majority of clients who subscribed to a term deposit, but with a relatively high number of false positives. This makes it a useful baseline model and highlights clear opportunities for further improvement.
