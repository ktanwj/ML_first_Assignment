from __future__ import print_function

# configs
SEED_NUM = 999
OUTPUT_FOLDER_PATH = './output/'
DATA_FOLDER_PATH = './data/'
BANK_FOLDER_PATH = DATA_FOLDER_PATH + 'bank/'
WEATHER_FOLDER_PATH = DATA_FOLDER_PATH + 'weather/'

# import core packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle

from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error
from imblearn.under_sampling import NearMiss

# sklearn modules
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score, roc_curve, roc_auc_score, auc

# sklearn models
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPClassifier

# Dim. Red. packages
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import locally_linear_embedding

# parallel computing packages
from itertools import product
from joblib import Parallel, delayed

import warnings
import time
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn' - to deal with SettingWithCopyWarning in Pandas

# helper functions
def get_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def put_pickle_file(filepath, file):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)


def encoder(df):
    labelencoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtypes=='object':
            df[col]=labelencoder.fit_transform(df[col])
    return df

def prep_bank_data(final_samples=1500, path=BANK_FOLDER_PATH+'train_data.csv'):
    # encodings_to_try = ['utf-8', 'ISO-8859-1']
    # for encoding in encodings_to_try:
    #     try:
    #         df = pd.read_csv(path, encoding=encoding)
    #         print("File successfully read with encoding:", encoding)
    #         break
    #     except UnicodeDecodeError:
    #         print("Failed to read with encoding:", encoding)
    df = pd.read_csv(path)
    df = encoder(df)
    scaler = StandardScaler()
    
    X = df.drop(columns=['Exited', 'id','CustomerId', 'Surname'])
    X = scaler.fit_transform(X)
    y = df[['Exited']].astype('int')

    # define the undersampling method
    undersample = NearMiss(version=1, n_neighbors=3)
    X, y = undersample.fit_resample(X, y)

    # randomly sample rows to reduce data size
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    indices = indices[:final_samples]
    # Choose the same indices for both x and y
    X = X[indices]
    y = y.iloc[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = SEED_NUM) # (only used when there is only one dataset)
    put_pickle_file(BANK_FOLDER_PATH+'X_train.pkl', X_train)
    put_pickle_file(BANK_FOLDER_PATH+'X_test.pkl', X_test)
    put_pickle_file(BANK_FOLDER_PATH+'y_train.pkl', y_train)
    put_pickle_file(BANK_FOLDER_PATH+'y_test.pkl', y_test)

    return X_train, X_test, y_train, y_test

def prep_weather_data(final_samples=1500, path = WEATHER_FOLDER_PATH + 'processed_weather.csv'):
    df = pd.read_csv(path)
    df = df.iloc[:,2:]
    df.drop(['index.2','index.1','Date'], axis=1, inplace=True)

    features=['month', 'Rainfall', 'Rainfall_prev', 'Evaporation', 'Evaporation_prev', 
          'Sunshine', 'Sunshine_prev', 'WindGustSpeed', 'WindGustSpeed_prev',
           'Humidity9am', 'Humidity9am_prev', 'Cloud9am', 'Cloud9am_prev', 'AverageTemp', 'AverageTemp_prev',
           'WindGustDir_N', 'WindGustDir_S', 'WindGustDir_E', 'WindGustDir_W',
          'Location_Changi', 'Location_Sentosa', 'Location_Tuas', 'Location_Woodlands',
          'Pressure9am_HIGH', 'Pressure9am_LOW', 'Pressure9am_MED']

    y = df["RainTomorrow"]
    X = df[features]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # define the undersampling method
    undersample = NearMiss(version=1, n_neighbors=3)
    X, y = undersample.fit_resample(X, y)

    # randomly sample rows to reduce data size
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    indices = indices[:final_samples]
    X = X[indices]
    y = y.iloc[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = SEED_NUM) # (only used when there is only one dataset)
    put_pickle_file(WEATHER_FOLDER_PATH+'X_train.pkl', X_train)
    put_pickle_file(WEATHER_FOLDER_PATH+'X_test.pkl', X_test)
    put_pickle_file(WEATHER_FOLDER_PATH+'y_train.pkl', y_train)
    put_pickle_file(WEATHER_FOLDER_PATH+'y_test.pkl', y_test)

    return X_train, X_test, y_train, y_test


"""
Helper functions
"""
def plot_learning_curve(model_name, best_model, X, y, score, data_name, dim_red, cv=5):
    total_sample_size = X.shape[0]
    # Plot the learning curve
    train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=5, scoring=score, n_jobs=-1)

    # Calculate the mean and standard deviation of the training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    train_sample_proportion = train_sizes/total_sample_size*100
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sample_proportion, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.fill_between(train_sample_proportion, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.plot(train_sample_proportion, train_scores_mean, 'o-', color="r", label="Training F1 Score")
    plt.plot(train_sample_proportion, test_scores_mean, 'o-', color="g", label="Validation F1 Score")
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('F1 Score')
    plt.title(f'{model_name} learning Curve on {data_name} using {dim_red}')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'src/{data_name}_output/learning_curve/{model_name}_learning_curve_{dim_red}.png')
    plt.show()

def plot_roc(model_name, best_model, X_test, y_test, data_name, dim_red):
    # Predict probabilities for the test data
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Calculate the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve on {data_name} using {dim_red}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'output/{model_name}_{data_name}_ROC_{dim_red}.png')

def base_model_runs(X_train, y_train, X_test, y_test):
    algorithms = [MLPClassifier()]
    algorithm_names = ["MLPClassifier"]
    # main hyperparameter tuning step
    for i in range(0, len(algorithms)):
        start_time = time.process_time()
        print("################   %s   ################" %algorithm_names[i])
        clf = algorithms[i]
        clf.fit(X_train, y_train)
        end_time = time.process_time()
        print(f'fit time: {round(end_time - start_time,3)}')
        y_pred = clf.predict(X_test)
        score = f1_score(y_test, y_pred)
        print(f'f1_score: {round(score,3)}')


def model_tuning(X_train, y_train, X_test, y_test, scores, data_name, dim_red, verbose = False):
    
    # initialisation
    train_report_column_names = ['model', 'Scoring metric', 'Hyperparameters', 'Mean score (validation)', 'Std score (validation)', 'Mean wall clock time']
    validation_report_column_names = ['model', 'Score', 'Best hyperparameters', 'Test score', 'Best refit time']

    train_validation_report = pd.DataFrame(columns=train_report_column_names)
    test_report = pd.DataFrame(columns=validation_report_column_names)

    # Parameters to be tuned. 
    tuned_parameters = [
                        [{'hidden_layer_sizes': [(100,), (100,50), (100,25), (100,50,25)], 'activation': ['tanh'], 'learning_rate_init': [0.01,0.05], 'learning_rate': ['constant', 'adaptive']}]
                        ]
    
    algorithms = [MLPClassifier()]
    algorithm_names = ["MLPClassifier"]

    # main hyperparameter tuning step
    for i in range(0, len(algorithms)):
        print("################   %s   ################" %algorithm_names[i])
        # scores = ['accuracy','precision_macro','recall_macro', 'f1']
        model_list = []
        score_list = []
        mean_score_list = []
        std_score_list = []
        params_list = []
        wall_clock_list = []
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(algorithms[i], tuned_parameters[i], cv=5,
                            scoring='%s' % score)
            clf.fit(X_train, y_train)
            if verbose:
                print("Best parameters set found on development set:")
                print()
                print(clf.best_params_)
                print()
                print("Grid scores on development set:")
                print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            wall_clock_times = clf.cv_results_['mean_fit_time']
            for mean, std, wall_clock_time, params in zip(means, stds, wall_clock_times, clf.cv_results_['params']):
                model_list.append(algorithm_names[i])
                score_list.append(score)
                mean_score_list.append(round(mean,3))
                std_score_list.append(round(std,3))
                params_list.append(params)
                wall_clock_list.append(wall_clock_time)
                if verbose:
                    print("%0.3f (+/-%0.03f) for %r"
                    % (mean, std * 2, params))
            if verbose:
                print()

                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
            y_true, y_pred = y_test, clf.predict(X_test)

            if verbose:
                print(classification_report(y_true, y_pred))
                print("Detailed confusion matrix:")
                print(confusion_matrix(y_true, y_pred))
                print("Precision Score: \n")
                print(precision_score(y_true, y_pred))
                print()
            
            # generate test report
            test_df = pd.DataFrame(np.column_stack([algorithm_names[i], score, clf.best_params_, clf.score(X_test, y_test), clf.refit_time_]), columns=validation_report_column_names)
            test_report = pd.concat([test_report, test_df])
            
            # plot learning curve
            plot_learning_curve(model_name=algorithm_names[i], best_model=clf.best_estimator_, X = X_train, y=y_train, data_name=data_name, dim_red = dim_red, score=score)

            # plot ROC curve
            plot_roc(model_name=algorithm_names[i], best_model=clf, X_test=X_test, y_test=y_test, data_name=data_name, dim_red=dim_red)

            # save classification report
            # Generate the classification report
            report = classification_report(y_true, y_pred)

            # Specify the file path where you want to save the report
            file_path = f"src/{data_name}_output/classification_report/{algorithm_names[i]}_classification_report_{dim_red}.txt"

            # Write the classification report to the file
            with open(file_path, "w") as file:
                file.write(report)
        
        # generate train_validation_report
        report_df = pd.DataFrame(np.column_stack([model_list, score_list, params_list, mean_score_list, std_score_list, wall_clock_list]), columns=train_report_column_names).sort_values(by='Mean score (validation)', ascending=False).head()
        train_validation_report = pd.concat([train_validation_report, report_df])

        # save train_report
        train_validation_report.to_csv(f'src/{data_name}_output/model_perf/{algorithm_names[i]}_train_validation_report_{dim_red}.csv')

        # save test_report
        test_report.to_csv(f'src/{data_name}_output/model_perf/{algorithm_names[i]}_test_report_{dim_red}.csv')
    return train_validation_report, test_report