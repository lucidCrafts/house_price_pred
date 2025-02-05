# Housing Price Prediction with Gradient Boosting Regressor

This project implements a housing price prediction model using the Gradient Boosting Regressor. The goal is to predict the sale price of houses based on various features using a dataset (CSV file). The process involves data preprocessing, model training, and model evaluation.

## Project Overview

This repository contains the necessary code to:
1. **Load and preprocess data** – The dataset is cleaned, missing values are handled, and irrelevant columns are dropped.
2. **Feature transformation** – Numerical and categorical features are preprocessed using scaling and encoding techniques.
3. **Model training** – A Gradient Boosting Regressor is used to train the model, with hyperparameter optimization performed using GridSearchCV.
4. **Model evaluation** – The model's performance is evaluated using metrics like MAE, MSE, RMSE, and R² score.

## Files

### `housing_price_prediction.py`
Contains the code for loading, preprocessing data, model training, and evaluation.

- **load_and_preprocess_data(file_path)**: This function loads the dataset and preprocesses it by imputing missing values, dropping unnecessary columns, and cleaning column names.
- **create_preprocessing_pipeline(numerical_cols, categorical_cols)**: This function creates a preprocessing pipeline for numerical and categorical columns.
- **split_data(df, numerical_cols, categorical_cols)**: Splits the data into training and testing datasets.
- **grid_search_evaluate(X_train, y_train, X_test, y_test, preprocessor)**: Performs grid search on hyperparameters, fits the model, and evaluates the results using metrics like MAE, MSE, RMSE, and R² score.

## Requirements

To run this project, the following libraries are required:

- pandas
- numpy
- scikit-learn

You can install them using `pip`:

```bash
pip install pandas numpy scikit-learn
