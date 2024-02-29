"""Machine Learning model for predicting survivors of Titanic's disaster"""
import pandas as pd
from kaggle.titanic.functions import download_titanic_data
#from kaggle.titanic.preprocessing import Preprocessor
from kaggle.titanic.preprocessing2 import Preprocessor
from kaggle.titanic.functions import LOCAL_PATH, TITANIC_PATH
from scipy.stats import expon, loguniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

train_data, test_data = download_titanic_data()

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

titanic_data = train_data.drop("Survived", axis=1)
titanic_labels = train_data["Survived"].copy()

preprocessor = Preprocessor()
X_train = preprocessor.fit_transform(titanic_data)
y_train = titanic_labels

param_distribs = {
        'svc__kernel': ['linear', 'rbf'],
        'svc__C': loguniform(20, 200_000),
        'svc__gamma': expon(scale=1.0),
    }

svc_pipeline = Pipeline([("preprocessor", preprocessor), ("svc", SVC())])
rnd_search = RandomizedSearchCV(svc_pipeline,
                                param_distributions=param_distribs,
                                n_iter=10, cv=3,
                                scoring='accuracy',
                                verbose=2,
                                random_state=42)

rnd_search.fit(titanic_data, titanic_labels)

rnd_search.best_params_

selector_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectFromModel(RandomForestRegressor(random_state=42),
                                 threshold=0.005)),  # min feature importance
    ('svc', SVC(C=rnd_search.best_params_["svc__C"],
                gamma=rnd_search.best_params_["svc__gamma"],
                kernel=rnd_search.best_params_["svc__kernel"])),
])

selector_accura = cross_val_score(selector_pipeline,
                                  titanic_data,
                                  titanic_labels,
                                  scoring="accuracy",
                                  cv=3)
pd.Series(selector_accura).describe()

final_model = rnd_search.best_estimator_
final_predictions = final_model.predict(test_data)
final_predictions

#
# Saving the final predictions to a csv
#
# Convert final_predictions to a DataFrame
df = pd.DataFrame(final_predictions)

# Rename the columns
df.columns = ['Survived']

# Add a new column 'Id' with values from the index
df['PassengerId'] = test_data.index

# Reorder columns
df = df[['PassengerId', 'Survived']]

save_file_in = LOCAL_PATH + TITANIC_PATH

df.to_csv(save_file_in + '/predicted_results.csv', index=False)