import os


def file_exists(path):
    return os.path.isfile(path)


def csv(df, file_w_path):
    if file_exists(file_w_path):
        print(f"File {file_w_path} already exists")
    else:
        df.to_csv(file_w_path)


def append_csv(df, file_w_path):
    if file_exists(file_w_path):
        df.to_csv(file_w_path, mode='a', header=False)
    else:
        print(f"File {file_w_path} doesn't exist")


def create_dir(dir_name):
    dir_path = get_results_dir() + dir_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory {dir_path} already exists")
    return dir_path


def get_results_dir():
    return os.getcwd()+'/scripts/risk_pipeline/outputs/results/'
