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