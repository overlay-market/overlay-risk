import os
from pathlib import Path

DATA = 'data'
DATA_EXT = '.csv'
CACHE = 'cache'


def file_exists(path):
    return os.path.isfile(path)


def csv(df, file_w_path):
    if file_exists(file_w_path):
        print(f"File {file_w_path} already exists")
    else:
        df.to_csv(file_w_path)


def create_dir(dir_name):
    dir_path = get_results_dir() / dir_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory {dir_path} already exists")
    return dir_path


def get_data_dir():
    return Path(os.path.dirname(os.path.realpath(__file__))).parent / DATA


def get_results_dir():
    return Path(os.path.dirname(os.path.realpath(__file__))) / CACHE


def get_paths(filename):
    filepath = str(get_data_dir() / filename) + DATA_EXT  # datafile
    resultsname = filename.replace('_treated', '')
    resultspath = get_results_dir() / resultsname
    return filepath, resultsname, resultspath
