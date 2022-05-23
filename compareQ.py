import datetime
import datetime
from es import fetch
from plotQOnGivenDay import plotQs
import sys
import os

args = sys.argv
splitBySlash = args[1].split("/")

dt = datetime.datetime(int(splitBySlash[0]), int(splitBySlash[1]), int(splitBySlash[2]))
fileName = f"docs_{dt.year}{str(dt.month).zfill(2)}{str(dt.day).zfill(2)}.pickle"
filePath = f"{os.getcwd()}/pickles/{fileName}"
fetch.fetchDocsByDatetime(dt, filePath)
plotQs(filePath)
