import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
from sklearn.preprocessing import LabelEncoder

mx_l, mn_l = 0, 0


def fill_with_random(new_df):
    """Function to fill the missing values using random imputation method."""
    for col in new_df.columns:
        if new_df[col].isna().sum() > 0:
            new_df[col] = new_df[col].apply(lambda x: np.random.choice(new_df[col].dropna().values) if pd.isna(x) else x)

    return new_df


def min_max_normalize(x_i: np.ndarray) -> np.ndarray:
    """Function to normalize the input using min-max method."""
    mx_val, mn_val = np.max(x_i), np.min(x_i)

    return (x_i - mn_val) / (mx_val - mn_val)


def reverse_normalization(x_i: np.ndarray, mx_x: float, mn_x: float) -> np.ndarray:
    """Function to reverse the normalization."""
    x_i *= (mx_x - mn_x)
    return x_i + mn_x


def get_data() -> tuple:
    """Function to load and return the dataset in a proper form to use later."""
    global mx_l, mn_l
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')
    df = pd.read_csv(os.path.join(dataset_path, 'Life Expectancy Data.csv'))
    print(f"The dataframe has the following missing values: {df.isnull().sum()}")
    df = fill_with_random(df)
    print(f"The missing values have been filled using random replacement: {df.isnull().sum()}")

    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    print(f"The Country column has become numerical using LabelEncoder!")

    df = pd.get_dummies(df, columns=['Status'], drop_first=True)
    print(f"The Status column has become numerical using One-hot encoding!")

    mx, mn = df['Year'].max(), df['Year'].min()
    mx_l, mn_l = df['Life expectancy '].max(), df['Life expectancy '].min()

    split_point = (2010 - mn) / (mx - mn)

    df = df.apply(min_max_normalize, axis=0)
    print("Normalization is done!")

    df_train = df[df['Year'] <= split_point]
    df_test = df[df['Year'] > split_point]

    y_train, y_test = df_train['Life expectancy '].to_numpy(), df_test['Life expectancy '].to_numpy()
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # y_train = reverse_normalization(y_train, mx_l, mn_l)
    # y_test = reverse_normalization(y_test, mx_l, mn_l)

    X_train, X_test = df_train.drop(columns=['Life expectancy ']).to_numpy(), df_test.drop(columns=['Life expectancy ']).to_numpy()
    print("Splitting the dataset into train and test set is finished!")

    return X_train, X_test, y_train, y_test
