import os


def get_file_name_by_datetime(dt):
    return f"docs_{dt.year}{str(dt.month).zfill(2)}{str(dt.day).zfill(2)}"


def get_json_file_path_by_datetime(dt):
    fileName = get_file_name_by_datetime(dt)
    return f"{os.getcwd()}/jsons/{fileName}.json"
