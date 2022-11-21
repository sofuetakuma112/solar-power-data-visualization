import os


def getPickleFileNameByDatetime(dt):
    return f"docs_{dt.year}{str(dt.month).zfill(2)}{str(dt.day).zfill(2)}.pickle"


def getPickleFilePathByDatetime(dt, data_type = "raw"):
    fileName = getPickleFileNameByDatetime(dt)
    return f"{os.getcwd()}/pickles/{data_type}/{fileName}"


def getPickleFilePath(fileName, data_type = "raw"):
    return f"{os.getcwd()}/pickles/{data_type}/{fileName}"
