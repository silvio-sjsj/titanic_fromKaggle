### Kaggle-titanic

This repository contain my attempts on exploring and predicting surviving passengers using the Titanic dataset.
This is a commom dataset for studying and learning on how to explore and analysis datasets as well as to build Machine Learning models.

The folders will contain attempts named "attempt_x" where x is a tentative, starting from 1.
These attempts are results of my studies on Machine Learning models and came to me by a variety of directions like youtube channels, hints on how to start exploring datasets beyond text-books exercises etc.
In particular the first attempt is a result of studying the textbook "Hands-on Machine Learning" by Aurélien Géron.

It contains most of what I learned so far, which is basically a little of feature engeneering, model selection, a little on how to tune the model and so on.

The competition [homepage on Kaggle](http://www.kaggle.com/c/titanic-gettingStarted).

### Attempt 1:

I will do just a little of feature engeneering here and try a RandomizedSearchCV for a possible better solution.
This is my first attempt on the titanic dataset and my first kaggle challenge.
I will be back on this dataset in the future to see how can I make better attempts on solving this challenge.

I hope this repository can be useful for someone who is starting their studies in Machine Learning, just as other people's notebooks  and repositories were useful for my own learning.

**My notebook on Kaggle can be found** [here](https://www.kaggle.com/code/silviosjsj/titanic-attempt-1-with-details)  
Any critics, suggestion, comments etc will be appreciated.

**What you'll find in the files:**
 * functions.py: functions for downloading the data from the internet, saving figures locally, Cramer's V correlation and other functions used in the project.
 * titanic_exploring_data.py: the data exploration code with graphs, correlations and other exploratory stuff.
 * titanic_model.py: the code to fit the model
 * preprocessing.py: the complete preprocessing pipeline used in the project for transforming and preparing the data.
 * titanic.tgz: .tgz files containing the test and train data as provided by Kaggle.
 * predicted_results.csv: the predicted results submitted to Kaggle.
 * the `images` folder contain all the images generated in the above codes
 * the `titanic` folder contains the test.csv and train.csv data provided by Kaggle.

#### Goal of this first attempt:
The goal of this first attempt was to build a simple model for predicting survivors of the Titanic disaster and to learn something in the way.
The goal was to use simple data analysis, some feature engeneering to go beyond the features already given in the dataset and some data exploration.

The code will show basic examples of **data handling**, like how to download and import data locally with pandas, cleaning datas and some exploration using Matplotlib and Seaborn, and feature engeneering;
**data analysis**, **preprocessing** with the creation of a simple but complete pipeline to tranform the data and let it ready to Machine Learning models; **modelling** and **evaluation**.

#### Results
In this attempt we used a SVC with a RandomizedSeachCV and we selected the most important attributes through SelectFromModel based on a RandomForestRegressor.

We used few parameters from the parameter space for the RandomizedSearchCV for the sake of computational time, and 3 folds with 10 iterations each, mostly for computational time.

The final result was not the best, but good enough: the best estimator gave an accuracy score of 0.82 between the folds.
It is a pretty good result as a first exercise and without some features that can impact the result even further (Cabin and Name), and the features created, like the age interval,
were based on a very few assumptions and can be increased in rigor in future attempts.

### Attempt 2
