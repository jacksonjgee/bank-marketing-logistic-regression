## Project Overview

This project investigates the UCI Bank Marketing dataset, which contains information from direct marketing campaigns conducted by a Portuguese banking institution.

The goal is to build a machine learning model that predicts whether a customer will subscribe to a term deposit.

## Research Question

> Can customer characteristics and previous marketing campaign information be used to predict whether a customer subscribes to a term deposit?

## Dataset Soruce & Info

The dataset was sourced from:
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). 

The dataset is a multivariable dataset with 45211 instances and 16 features.

## Dataset Features

| Variable      | Type          | Description                                         |
| ------------- | ------------- | --------------------------------------------------- |
| `age`         | Numerical     | Client's age                                        |
| `job`         | Categorical   | Client's occupation                                 |
| `marital`     | Categorical   | Marital status                                      |
| `education`   | Categorical   | Education level                                     |
| `default`     | Binary        | Whether the client has credit in default            |
| `balance`     | Numerical     | Average yearly account balance                      |
| `housing`     | Binary        | Whether the client has a housing loan               |
| `loan`        | Binary        | Whether the client has a personal loan              |
| `contact`     | Categorical   | Contact communication method                        |
| `day_of_week` | Categorical   | Day of the week of the last contact                 |
| `month`       | Categorical   | Month of the last contact                           |
| `duration`    | Numerical     | Duration of the last contact in seconds             |
| `campaign`    | Numerical     | Number of contacts made during the current campaign |
| `pdays`       | Numerical     | Days since the client was previously contacted      |
| `previous`    | Numerical     | Number of contacts made before the current campaign |
| `poutcome`    | Categorical   | Outcome of the previous marketing campaign          |
| `y`           | Binary target | Whether the client subscribed to a term deposit     |
