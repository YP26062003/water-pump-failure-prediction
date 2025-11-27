import pandas as pd
import numpy as np

# Function to handle datetime features
def handle_dates(train_data, test_data):
    # Convert 'date_recorded' to datetime
    train_data["date_recorded"] = pd.to_datetime(train_data["date_recorded"], format="%Y-%m-%d")
    test_data["date_recorded"] = pd.to_datetime(test_data["date_recorded"], format="%Y-%m-%d")

    # Calculate the first recorded date from the training set
    first_recorded_date = train_data["date_recorded"].min()

    # Convert dates to the number of days since the first recorded date
    train_data["date_recorded"] = (train_data["date_recorded"] - first_recorded_date).dt.days
    test_data["date_recorded"] = (test_data["date_recorded"] - first_recorded_date).dt.days

    return train_data, test_data

# Function to find missing categorical columns
def find_missing_values(train_competition_input):
    missing_values = train_competition_input.isnull().sum()
    print(missing_values[missing_values > 0])
    missing_columns = missing_values[missing_values > 0].index
    categorical_cols = train_competition_input.select_dtypes(include=["object"], exclude=["int"]).columns
    missing_cat_columns = missing_columns[missing_columns.isin(categorical_cols)]
    return missing_cat_columns

def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def handle_coordinates(train_competition_input, test_competition_input):
    # Calculate mean latitude and longitude
    mean_latitude = train_competition_input['latitude'].mean()
    mean_longitude = train_competition_input['longitude'].mean()

    # Add distance feature to your dataset
    train_competition_input['distance_to_mean'] = train_competition_input.apply(
        lambda row: euclidean_distance(row['latitude'], row['longitude'], mean_latitude, mean_longitude), axis=1)
    test_competition_input['distance_to_mean'] = test_competition_input.apply(
        lambda row: euclidean_distance(row['latitude'], row['longitude'], mean_latitude, mean_longitude), axis=1)

    # Drop longitude and latitude columns (optional)
    train_competition_input = train_competition_input.drop(columns=['longitude', 'latitude'])
    test_competition_input = test_competition_input.drop(columns=['longitude', 'latitude'])
    return train_competition_input, test_competition_input


def drop_columns(train_competition_input, test_competition_input):
    train_competition_input.drop(columns=['id'], inplace=True)
    test_competition_input.drop(columns=['id'], inplace=True)

    low_cardinality_cols = [col for col in train_competition_input.columns if
                            train_competition_input[col].nunique() == 1]
    train_competition_input.drop(columns=low_cardinality_cols, inplace=True)
    test_competition_input.drop(columns=low_cardinality_cols, inplace=True)

    duplicate_columns = train_competition_input.columns[train_competition_input.T.duplicated()]
    train_competition_input.drop(columns=duplicate_columns, inplace=True)
    test_competition_input.drop(columns=duplicate_columns, inplace=True)

    return train_competition_input, test_competition_input
