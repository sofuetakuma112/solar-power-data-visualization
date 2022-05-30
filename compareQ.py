import datetime
import datetime
from es import fetch
from plotQOnGivenDay import plotQs
from time_diff_in_actual_Q import timeDiffInActualQ
from utils.file import getPickleFilePathByDatetime
import sys

args = sys.argv
splitBySlash = args[1].split("/")

dt = datetime.datetime(int(splitBySlash[0]), int(splitBySlash[1]), int(splitBySlash[2]))
filePath = getPickleFilePathByDatetime(dt)

fetch.fetchDocsByDatetime(dt)
plotQs(filePath)
# timeDiffInActualQ(filePath)
