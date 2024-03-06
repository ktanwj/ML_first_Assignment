from src.utils import *
from src.utils_ed import *
import six
import sys
sys.modules['sklearn.externals.six'] = six

import mlrose_hiive as mlrose
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import NearMiss
import numpy as np
import pandas as pd


# data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# sklearn models
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from __future__ import print_function

import warnings
import time
warnings.filterwarnings("ignore")
%matplotlib inline
SEED_NUM = 42

# CONFIG
X_TRAIN_PATH = 'src/section2/data/bank/x_train.pkl'
X_TEST_PATH = 'src/section2/data/bank/x_test.pkl'
Y_TRAIN_PATH = 'src/section2/data/bank/y_train.pkl'
Y_TEST_PATH = 'src/section2/data/bank/y_test.pkl'
X_train = get_pickle_file(X_TRAIN_PATH)
X_test = get_pickle_file(X_TEST_PATH)
y_train = get_pickle_file(Y_TRAIN_PATH)
y_test = get_pickle_file(Y_TEST_PATH)
model_tuning(X_train, y_train, X_test, y_test, scores=['f1'], data_name='', verbose = False)


def model_tuning(X_train, y_train, X_test, y_test, scores, data_name, verbose = False):
    
    # initialisation
    train_report_column_names = ['model', 'Scoring metric', 'Hyperparameters', 'Mean score (validation)', 'Std score (validation)', 'Mean wall clock time']
    validation_report_column_names = ['model', 'Score', 'Best hyperparameters', 'Test score', 'Best refit time']

    train_validation_report = pd.DataFrame(columns=train_report_column_names)
    test_report = pd.DataFrame(columns=validation_report_column_names)

    # Parameters to be tuned. 
    tuned_parameters = [[{'algorithm':['random_hill_climb'], 'learning_rate': [0.01, 0.05, 0.1],'max_attempts': [50, 100], 'restarts':[0, 3]}],
                        [{'algorithm':['simulated_annealing'], 'learning_rate': [0.01, 0.05, 0.1],'max_attempts': [50, 100], 'restarts':[0, 3]}],
                        [{'algorithm':['genetic_alg'], 'learning_rate': [0.01, 0.05, 0.1],'max_attempts': [50, 100], 'restarts':[0, 3]}]]
    algorithms = [mlrose.NeuralNetwork(hidden_nodes=[256], activation='relu',
                                        max_iters=1000, bias=True, is_classifier=True, 
                                        early_stopping=False, clip_max=5, curve=True),
                  mlrose.NeuralNetwork(hidden_nodes=[256], activation='relu',
                                        max_iters=1000, bias=True, is_classifier=True, 
                                        early_stopping=False, clip_max=5, curve=True),
                  mlrose.NeuralNetwork(hidden_nodes=[256], activation='relu',
                                        max_iters=1000, bias=True, is_classifier=True, 
                                        early_stopping=False, clip_max=5, curve=True)]
    algorithm_names = ["random_hill_climb","simulated_annealing","SVC","genetic_alg"]
    
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
            plot_learning_curve(model_name=algorithm_names[i], best_model=clf.best_estimator_, X = X_train, y=y_train, data_name=data_name, score=score)

            # plot ROC curve
            plot_roc(model_name=algorithm_names[i], best_model=clf, X_test=X_test, y_test=y_test, data_name=data_name)

            # save classification report
            # Generate the classification report
            report = classification_report(y_true, y_pred)

            # Specify the file path where you want to save the report
            file_path = f"src/section2/classification_report/{algorithm_names[i]}_classification_report.txt"

            # Write the classification report to the file
            with open(file_path, "w") as file:
                file.write(report)
        
        # generate train_validation_report
        report_df = pd.DataFrame(np.column_stack([model_list, score_list, params_list, mean_score_list, std_score_list, wall_clock_list]), columns=train_report_column_names).sort_values(by='Mean score (validation)', ascending=False).head()
        train_validation_report = pd.concat([train_validation_report, report_df])

        # save train_report
        train_validation_report.to_csv(f'src/section2/model_perf/{algorithm_names[i]}_train_validation_report.csv')

        # save test_report
        test_report.to_csv(f'src/section2/model_perf/{algorithm_names[i]}_test_report.csv')

def plot_validation_curve(data_name, title, x_lab, y_lab, model_name, model, score_metric, param_name, param_range):
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name=param_name, param_range=param_range,
        scoring=score_metric, cv=5, n_jobs=-1)
    
    # Calculate the mean and standard deviation of the scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    best_test_score = np.max(test_scores_mean)
    best_test_index = np.argmax(test_scores_mean)

    if type(param_range[0]) is tuple:
        param_range = np.array([str(t) for t in param_range])
        
    # Plot the validation curve
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ticklabel_format(style='scientific', axis='both')
    plt.ylabel(y_lab)
    plt.plot(param_range, train_scores_mean, label="Training Score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Validation Score", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.axvline(x=param_range[best_test_index], color='r', linestyle='--')
    plt.annotate(f'Optimal {param_name}: {param_range[best_test_index]}', (param_range[best_test_index], best_test_score), xytext=(10, 20),
            textcoords='offset points', arrowprops={'arrowstyle': '->', 'color': 'r'})

    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'src/section2/validation_curve/{model_name}_{param_name}_{score_metric}.png')
    plt.close()


def plot_learning_curve(model_name, best_model, X, y, score, data_name, cv=5):
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
    plt.title(f'{model_name} learning Curve with GridSearchCV')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'src/section2/learning_curve/{model_name}_learning_curve.png')
    plt.show()

def plot_roc(model_name, best_model, X_test, y_test, data_name):
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
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'src/section2/ROC_curve/{model_name}_ROC.png')