import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

def read(fileName):
    return pd.read_csv(fileName)

def giveinfo(df):
    return df.info()

def NullValChecker(df):
    nullValPerColumn = {column: int(df[column].isnull().sum()) for column in df.columns}
    for column, null_count in nullValPerColumn.items():
        print(f"{column}: {null_count}")
    print(df.head())
    return nullValPerColumn

def NullColumnsRemover(df):
    return df.dropna()

def EncodingData(df, columnName):
    """
    Performs one-hot encoding on the specified column and ensures the dtype is np.float64.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columnName (str): The column to encode.

    Returns:
        pd.DataFrame: The encoded DataFrame with one-hot columns as np.float64.
    """
    encoded_df = pd.get_dummies(df, columns=[columnName])
    # Convert all encoded columns to np.float64
    for col in encoded_df.columns:
        if encoded_df[col].dtype == np.bool_:
            encoded_df[col] = encoded_df[col].astype(np.float64)
    return encoded_df

def scale_features(df):
    scaler = StandardScaler()
    
    # List of columns that should not be scaled
    columns_to_exclude = ['ocean_proximity_<1H OCEAN','ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN']

    # Select columns to scale (all columns except those in columns_to_exclude)
    columns_to_scale = [col for col in df.columns if col not in columns_to_exclude]
    
    # Scale the selected columns
    scaled_data = scaler.fit_transform(df[columns_to_scale])
    
    # Create a DataFrame with scaled data and the same columns
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
    
    # Add the non-scaled columns back to the scaled DataFrame
    for col in columns_to_exclude:
        scaled_df[col] = df[col].values
    
    # Ensure that the original column order is preserved
    return scaled_df[df.columns]



def train_test_split_and_save(df, feature_columns, target_column, test_size=0.20, shuffle=True, output_prefix=''):
    """
    Splits the dataset into training and testing sets, joins the feature and target data, 
    and saves them as separate CSV files.
    """
    # Splitting the dataset
    X = df[feature_columns]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    
    # Joining feature and target data
    train_data = X_train.join(y_train)
    test_data = X_test.join(y_test)
    
    # Saving to CSV files
    df.to_csv(f'{output_prefix}housing.csv', index=False)
    train_data.to_csv(f'{output_prefix}train_data.csv', index=False)
    test_data.to_csv(f'{output_prefix}test_data.csv', index=False)
    
    print(f"Files saved: {output_prefix}housing.csv, {output_prefix}train_data.csv, {output_prefix}test_data.csv")

# Main Workflow
df = read("housing.csv")
giveinfo(df)

# Removing null rows
df = NullColumnsRemover(df)

# One-hot encoding
df = EncodingData(df, 'ocean_proximity')

# Scaling features (excluding the target column)
df = scale_features(df)

# Train-test split
features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity_<1H OCEAN",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN"
]
target_column = "median_house_value"

# Split the data and save CSV files
train_test_split_and_save(df, features, target_column)
