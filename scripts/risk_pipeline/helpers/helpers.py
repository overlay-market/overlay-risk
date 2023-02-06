import os


def file_exists(path):
    return os.path.isfile(path)


def csv(df, file_w_path):
    if file_exists(file_w_path):
        print(f"File {file_w_path} already exists")
    else:
        df.to_csv(file_w_path)
