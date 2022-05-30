import os


def getPickleFileNameByDatetime(dt):
    return f"docs_{dt.year}{str(dt.month).zfill(2)}{str(dt.day).zfill(2)}.pickle"


def getPickleFilePathByDatetime(dt):
    fileName = getPickleFileNameByDatetime(dt)
    return f"{os.getcwd()}/pickles/{fileName}"


def getPickleFilePath(fileName):
    return f"{os.getcwd()}/pickles/{fileName}"
