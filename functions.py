"""Functions to load titanic dataset, save figures and also classes used for pipelines"""
import os
import urllib
import tarfile
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# Code to save the figures as high-res PNGs

IMAGES_PATH = Path()  / 'kaggle' / "titanic" / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Local variables
LOCAL_PATH = "kaggle/"
TITANIC_PATH = os.path.join("titanic")
TITANIC_URL = "https://github.com/ageron/data/raw/main/titanic.tgz"

def download_titanic_data(titanic_url=TITANIC_URL, titanic_path=TITANIC_PATH, local_path=LOCAL_PATH):
    """Download the titanic data and extract it to a csv"""
    if not os.path.isdir(local_path + titanic_path):
        os.makedirs(local_path + titanic_path)
    tgz_path = os.path.join(local_path + titanic_path, "titanic.tgz")
    urllib.request.urlretrieve(titanic_url, tgz_path)
    titanic_tgz = tarfile.open(tgz_path)
    titanic_tgz.extractall(path=local_path + "/titanic")
    titanic_tgz.close()
    return pd.read_csv(local_path + titanic_path + '/titanic/train.csv'), \
           pd.read_csv(local_path + titanic_path + '/titanic/test.csv')

#
# Correlation function - Cramer V
#
def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


#
# Complete Pipeline for treatment of the data
#
###############################################################################
class DropColumnsTransformer:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)
    
    def get_feature_names_out(self, input_features=None):
        return [col for col in input_features if col not in self.columns]

def sum_name(function_transformer, feature_names_in):
    return ["total_relatives"]  # feature names out

def sum_relatives(X):
    X_copy = X.copy()
    X_copy['total_relatives'] = X_copy['SibSp'] + X_copy['Parch']
    return X_copy[['total_relatives']]

total_relatives_pipeline = make_pipeline(
    FunctionTransformer(sum_relatives, feature_names_out=sum_name)
)

def cat_travel_name(function_transformer, feature_names_in):
    return ["traveling_category"]  # feature names out

def categorize_travel(X):
    X_copy = X.copy()
    
    conditions = [
        (X_copy['total_relatives'] == 0),
        (X_copy['total_relatives'] >= 1) & (X_copy['total_relatives'] <= 3),
        (X_copy['total_relatives'] >= 4)
    ]
    categories = ['A', 'B', 'C']

    X_copy['traveling_category'] = np.select(conditions, categories, default='Unknown')
    return X_copy[['traveling_category']]

travel_category_pipeline = make_pipeline(
    FunctionTransformer(categorize_travel, feature_names_out=cat_travel_name)
)

class_order = [[1, 2, 3]]

ord_pipeline = make_pipeline(
    OrdinalEncoder(categories=class_order)    
    )

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

#Numerical for fare
fare_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

def drop_transformer(X):
    X_copy = X.copy()
    X_copy = X_copy.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    return X_copy

def drop_columns():
    return make_pipeline(
        FunctionTransformer(drop_transformer)
    )

def interval_name(function_transformer, feature_names_in):
    return ["age_interval"]  # feature names out

def age_transformer(X):
    X_copy = X.copy()
    median_age_by_class = X_copy.groupby('Pclass')['Age'].median().reset_index()
    median_age_by_class.columns = ['Pclass', 'median_age']
    for index, row in median_age_by_class.iterrows():
        class_value = row['Pclass']
        median_age = row['median_age']
        X_copy.loc[X_copy['Pclass'] == class_value, 'Age'] = \
            X_copy.loc[X_copy['Pclass'] == class_value, 'Age'].fillna(median_age)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, np.inf]
    labels = ['(0, 10]', '(10, 20]', '(20, 30]', '(30, 40]', '(40, 50]', '(50, 60]', '(60, 70]', '(70, 100]']
    X_copy['age_interval'] = pd.cut(X_copy['Age'], bins=bins, labels=labels)
    return X_copy[['age_interval']]

def age_processor():
    return make_pipeline(
        FunctionTransformer(age_transformer, feature_names_out=interval_name),
        )

# Define your preprocessing steps as before
preprocessing_initial = ColumnTransformer([
        ("ord", ord_pipeline, ['Pclass']),
        ("age_processing", age_processor(), ['Pclass', 'Age']),
        ("num", fare_pipeline, ['Fare']),
        ("total_relatives", total_relatives_pipeline, ['SibSp', 'Parch'])],
        remainder='passthrough',
        verbose_feature_names_out=False
)

preprocessing_travel_category = ColumnTransformer(
    [("travel_category", travel_category_pipeline, ['total_relatives'])],
    remainder='passthrough',
    verbose_feature_names_out=False
)

preprocessing_cat = ColumnTransformer(
    [("cat", cat_pipeline, ['Sex', 'Embarked', 'traveling_category', 'age_interval'])],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# Define the columns you want to drop
columns_to_drop = ['Name', 'Cabin', 'Ticket']

# Create a transformer to drop those columns
preprocessing_drop_columns = DropColumnsTransformer(columns=columns_to_drop)

# Combine all transformers in a pipeline
preprocessor = make_pipeline(
    preprocessing_initial,
    preprocessing_travel_category,
    preprocessing_cat,
    preprocessing_drop_columns  # Add the transformer to drop columns at the end
)
###############################################################################