"""Exploring the data of the Kaggle's titanic dataset"""
import itertools
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kaggle.titanic.functions import cramers_v, save_fig, download_titanic_data, IMAGES_PATH, LOCAL_PATH, TITANIC_PATH, TITANIC_URL

sns.set_theme(style="whitegrid")
sns.set(font_scale = 1)

train_data, test_data = download_titanic_data()
train_data.head()

train_data.info()

train_data['PassengerId'].nunique()

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

train_data_copy = train_data.copy()

# Some histograms
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

train_data_copy.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")
plt.show()
#
#
#

train_data_copy['Sex'].value_counts()

median_age_by_class = train_data_copy.groupby('Pclass')['Age'].median().reset_index()
median_age_by_class.columns = ['Pclass', 'median_age']
median_age_by_class

for index, row in median_age_by_class.iterrows():
    class_value = row['Pclass']
    median_age = row['median_age']
    train_data_copy.loc[train_data_copy['Pclass'] == class_value, 'Age'] = train_data_copy.loc[train_data_copy['Pclass'] == class_value, 'Age'].fillna(median_age)

train_data_copy

train_data_copy['Cabin'].nunique()

train_data_copy['Cabin'].unique()

g = sns.catplot(data= train_data_copy,
                x= 'Fare',
                y= 'Embarked',
                col= 'Sex',
                hue='Pclass',
                alpha=.6, height=6
               )
save_fig("catplot_fare-embarked_per_sex-class")
g.despine(left=True)

g = sns.catplot(data= train_data_copy,
                x= 'Age',
                y= 'Embarked',
                col= 'Sex',
                hue='Pclass',
                alpha=.6, height=6
               )
save_fig("catplot_age-embarked_per_sex-class")
g.despine(left=True)

ax = plt.axes()
sns.heatmap(train_data_copy.corr(numeric_only=True), annot=True, ax = ax)
ax.set_title('Pearson correlation between features')
save_fig("Pearson_correlation_between_features")

train_data_copy['total_relatives'] = train_data_copy['Parch'] + train_data_copy['SibSp']
train_data_copy['total_relatives'].value_counts()

# Create a pivot table to count the number of survivors for each total number of relatives
survival_df = pd.pivot_table(train_data_copy, index='total_relatives', columns='Survived', aggfunc='size', fill_value=0)

# Add a column for the total number of passengers for each total number of relatives
survival_df['total_passengers'] = survival_df.sum(axis=1)

# Calculate the percentage of survivors for each total number of relatives
survival_df['survival_percentage'] = (survival_df[1] / survival_df['total_passengers']) * 100

# Rename the index and reset it to make it cleaner
survival_df.index.name = 'Total Relatives'
survival_df.reset_index(inplace=True)
survival_df

conditions = [
    (train_data_copy['total_relatives'] == 0),
    (train_data_copy['total_relatives'] >= 1) & (train_data_copy['total_relatives'] <= 3),
    (train_data_copy['total_relatives'] >= 4)
]
categories = ['A', 'B', 'C']

train_data_copy['traveling_category'] = np.select(conditions, categories, default='Unknown')
train_data_copy

bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]

train_data_copy['age_interval'] = pd.cut(train_data_copy['Age'], bins=bins)
train_data_copy

train_data_copy['age_interval'].value_counts()

# Create a pivot table to count the number of survivors for each total number of relatives
survival_df = pd.pivot_table(train_data_copy, index='age_interval', columns='Survived', aggfunc='size', fill_value=0)

# Add a column for the total number of passengers for each total number of relatives
survival_df['total_passengers'] = survival_df.sum(axis=1)

# Calculate the percentage of survivors for each total number of relatives
survival_df['survival_percentage'] = (survival_df[1] / survival_df['total_passengers']) * 100

# Rename the index and reset it to make it cleaner
survival_df.index.name = 'Age_Interval'
survival_df.reset_index(inplace=True)
survival_df

train_data_copy['age_interval'].value_counts()

#
# Calculate the correlation between features using Cramer V function
#
cols = list(train_data_copy.columns.drop('Name'))

corr_matrix = np.zeros((len(cols),len(cols)))

for col1, col2 in itertools.combinations(cols, 2):
    idx1, idx2 = cols.index(col1), cols.index(col2)
    corr_matrix[idx1, idx2] = cramers_v(pd.crosstab(train_data_copy[col1], train_data_copy[col2]))
    corr_matrix[idx2, idx1] = corr_matrix[idx1, idx2]

corr = pd.DataFrame(corr_matrix, index= cols, columns= cols)
fig, ax = plt.subplots(figsize= (10, 10))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables")
save_fig("Cramer_V_correlation")