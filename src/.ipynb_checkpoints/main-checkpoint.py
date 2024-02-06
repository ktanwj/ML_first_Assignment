# import libraries
import sys

# import ML libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# import helper functions
from utils import read_db_local_path, preprocess_data
from model import train_evaluate_models
from model_tuning import model_tuning

# ----------------------------------------------------------------------- #
# main script - run this
# ----------------------------------------------------------------------- #
def main():
    # reads file from DB stored locally
    df = read_db_local_path()

    # apply data pre-processing techniques to clean data 
    df = preprocess_data(df)
    
    # (not used in submission) re-run model_tuning.py (hyperparam tuning) if user prompts
    # if sys.argv[1] == 'yes':
    #     print('------------- Performing Hyperparameter tuning -------------')
    #     model_tuning()

    # train and evaluate models saved in model_config
    mapping = {'best_dt': DecisionTreeClassifier,
               'best_rf': RandomForestClassifier,
               'best_knn': KNeighborsClassifier,
               'best_xgb': XGBClassifier}

    train_evaluate_models(df, mapping)

if __name__ == "__main__":
    main()