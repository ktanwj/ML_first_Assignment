#####################
#  required modules #
#####################

# import libraries
import pandas as pd
import sqlite3
from .config import FILE_PATH

def read_db_local_path(file_path:str = FILE_PATH, test_connection:bool = True) -> pd.DataFrame:
    """ reads db file from the local path provided
    Parameters
    ----------
    file_path (optional) : str
        Path to the db file stored locally
    test_connection (optional) : bool
        True if we would like to test the connection
        
    Returns
    -------
    pd.DataFrame
        A pandas dataframe containing the returned file
    """
    conn = sqlite3.connect(file_path)
    df = pd.read_sql_query("SELECT * FROM fishing", conn)
    if test_connection:
        if df.empty:
            print('Imported file is empty. Please check.')
        else:
            print('Read file success!')
    return df

def check_no_dupes(df:pd.DataFrame):
    """ check pandas DataFrame for duplicated rows
    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
        
    Returns
    -------
    bool : True if no duplicates found, else False
    """
    check = df.isnull().sum().sort_values(ascending=False).sum() == 0
    if check:
        print('Test passed!')
    else:
        print('Test failed, please check!')