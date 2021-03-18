import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#data directory
data_dir = "./VFIB/data/adult/adult.csv"

# Columns with categorical variables
cat_columns = [
        "Workclass", "Education", "Country", "Relationship",
        "Martial Status", "Occupation", "Relationship",
        "Race", "Sex"
    ]

# All columns
columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status", \
              "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]

# Data types
types = {0: int, 1: str, 2: int, 3: str, 4: int, 5: str, 6: str, 7: str, 8: str, 9: str, 10: int,
                                11: int, 12: int, 13: str, 14: str}


# remove observations with missing feature values
def remove_missing(X):
    m = X.shape[0]
    print('Raw Dataset size : ', m)
    X.replace('nan', np.nan, inplace=True)
    X.dropna(inplace=True)
    n = X.shape[0]
    print('Size after dropping null values: ', n)
    print('Removed ', (m-n), ' observations')

    
# one-hot encode categorical data
def replace_categorical(X):
    X_sets = [X.select_dtypes(include=[i]).copy() for i in ['object', 'int']]
    X_cat , X_n = X_sets
    [print(n, ' set size:' , i.shape) 
     for (n,i) in zip(['\nCategorical',
                       'Continuous'],
                          X_sets)]
    
    X_cat = pd.get_dummies(X_cat, columns=cat_columns)
    return pd.concat([X_n, X_cat], axis=1)


# Split features and labels
def separate_label(X):
    y = X['Target'].copy()
    X = X.drop(['Target'], axis=1)
    y = LabelEncoder().fit_transform(y)
    return X, y


# Binarize continuous features with column mean
def binarize_features(X):
    for i in range(6):
        thresh = X.iloc[:, i].mean()
        X.iloc[:, i] = np.where(X.iloc[:, i].values > thresh, 1,0)
    return X


# Load and preprocess
def load_adult(binarize=False):
    data = pd.read_csv(
        data_dir,
        names=columns,
        sep=r'\s*,\s*',
        engine='python', skiprows=1,
        na_values="?",
        dtype=types)
    
    remove_missing(data)
    X, y = separate_label(data)
    X = replace_categorical(data)
    
    if binarize:
        binarize_features(X)

    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# Disparity amongst group outcomes
def p_rule(y_pred, x_sensitive, threshold=0.5):
    y_z_1 = y_pred[x_sensitive == 1] > threshold if threshold else y_pred[x_sensitive == 1]
    y_z_0 = y_pred[x_sensitive == 0] > threshold if threshold else y_pred[x_sensitive == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100
