import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.base import TransformerMixin
from sklearn import set_config

set_config(transform_output='pandas')

class Preprocessor(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        preprocessing_initial = self._initial_preprocessing()
        preprocessing_travel_category = self._travel_category_preprocessing()
        preprocessing_cat = self._categorical_preprocessing()
        preprocessing_drop_columns = self._drop_columns_preprocessing()

        # Combine all transformers in a pipeline
        preprocessor = make_pipeline(
            preprocessing_initial,
            preprocessing_travel_category,
            preprocessing_cat,
            preprocessing_drop_columns  # Add the transformer to drop columns at the end
        )

        return preprocessor.fit_transform(X)

    def _sum_name(self, function_transformer, feature_names_in):
        return ["total_relatives"]  # feature names out

    def _sum_relatives(self, X):
        X_copy = X.copy()
        X_copy['total_relatives'] = X_copy['SibSp'] + X_copy['Parch']
        return X_copy[['total_relatives']]

    def _total_relatives_pipeline(self):
        return make_pipeline(
            FunctionTransformer(self._sum_relatives, feature_names_out=self._sum_name)
        )

    def _cat_travel_name(self, function_transformer, feature_names_in):
        return ["traveling_category"]  # feature names out

    def _categorize_travel(self, X):
        X_copy = X.copy()
        
        conditions = [
            (X_copy['total_relatives'] == 0),
            (X_copy['total_relatives'] >= 1) & (X_copy['total_relatives'] <= 3),
            (X_copy['total_relatives'] >= 4)
        ]
        categories = ['A', 'B', 'C']

        X_copy['traveling_category'] = np.select(conditions, categories, default='Unknown')
        return X_copy[['traveling_category']]

    def _travel_category_pipeline(self):
        return make_pipeline(
            FunctionTransformer(self._categorize_travel, feature_names_out=self._cat_travel_name)
        )

    def _ordinal_encoder(self):
        class_order = [[1, 2, 3]]
        return make_pipeline(
            OrdinalEncoder(categories=class_order)
        )

    def _fare_pipeline(self):
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        )

    def _drop_transformer(self, X):
        X_copy = X.copy()
        X_copy = X_copy.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        return X_copy

    def _drop_columns(self):
        return make_pipeline(
            FunctionTransformer(self._drop_transformer)
        )

    def _interval_name(self, function_transformer, feature_names_in):
        return ["age_interval"]  # feature names out

    def _age_transformer(self, X):
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

    def _age_processor(self):
        return make_pipeline(
            FunctionTransformer(self._age_transformer, feature_names_out=self._interval_name)
        )

    def _initial_preprocessing(self):
        return ColumnTransformer([
                ("ord", self._ordinal_encoder(), ['Pclass']),
                ("age_processing", self._age_processor(), ['Pclass', 'Age']),
                ("num", self._fare_pipeline(), ['Fare']),
                ("total_relatives", self._total_relatives_pipeline(), ['SibSp', 'Parch'])],
                remainder='passthrough',
                verbose_feature_names_out=False
        )

    def _travel_category_preprocessing(self):
        return ColumnTransformer(
            [("travel_category", self._travel_category_pipeline(), ['total_relatives'])],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

    def _categorical_preprocessing(self):
        return ColumnTransformer(
            [("cat", self._categorical_pipeline(), ['Sex', 'Embarked', 'traveling_category', 'age_interval'])],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

    def _categorical_pipeline(self):
        return make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        )

    def _drop_columns_preprocessing(self):
        return self._drop_columns()


class DropColumnsTransformer:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)
    
    def get_feature_names_out(self, input_features=None):
        return [col for col in input_features if col not in self.columns]
