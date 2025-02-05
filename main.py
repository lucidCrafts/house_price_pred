import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Strip spaces from column names

    # Identifying numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Impute missing values
    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    # Drop unnecessary and high-missing columns
    df.drop(columns=[
        'Id', 'LotFrontage', 'Neighborhood', 'Alley', 'Exterior1st', 'Exterior2nd', 
        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 
        '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinType2', 'Heating', 'HeatingQC', 'Functional', 'BldgType', 'RoofMatl', 
        'RoofStyle', 'Condition1', 'Condition2', 'MiscFeature', 'Fence', 'PoolQC', 
        'ExterCond', 'Electrical', 'BsmtCond', 'Source'], inplace=True)

    df.dropna(inplace=True)  # Drop rows with missing values

    # Convert target column to integer
    df['SalePrice'] = df['SalePrice'].astype(int)

    return df

# Function to create preprocessing pipeline
def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

# Function to split data into train and test sets
def split_data(df, numerical_cols, categorical_cols):
    X = df[numerical_cols + categorical_cols]
    y = df['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to perform GridSearchCV and evaluate the model
def grid_search_evaluate(X_train, y_train, X_test, y_test, preprocessor):
    param_grid = {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7,10,30,50],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__subsample': [0.8, 1.0]
    }

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    # GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Cross-Validation Score:", best_score)

    # Get the best model and evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Main function to execute the entire workflow
def main(file_path):
    df = load_and_preprocess_data(file_path)

    # Define numerical and categorical columns
    numerical_cols = ['LotArea', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    categorical_cols = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                        'LandSlope', 'HouseStyle', 'SaleType', 'SaleCondition']

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, numerical_cols, categorical_cols)

    # Perform grid search and evaluate the model
    grid_search_evaluate(X_train, y_train, X_test, y_test, preprocessor)

# Execute the main function
if __name__ == '__main__':
    main('housing_dataset.csv')
