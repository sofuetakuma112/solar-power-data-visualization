import datetime

from utils.es import fetch
from plotQOnGivenDay import plotQs
from time_diff_in_actual_Q import timeDiffInActualQ
from utils.file import getPickleFilePathByDatetime
import sys

args = sys.argv
splitBySlash = args[1].split("/")
delay_str = args[2]
mode = args[3]

dt = datetime.datetime(int(splitBySlash[0]), int(splitBySlash[1]), int(splitBySlash[2]))
filePath = getPickleFilePathByDatetime(dt)

fetch.fetchDocsByDatetime(dt)

if mode == "q":
  plotQs(filePath, float(delay_str))
elif mode == "dt_diff":
  timeDiffInActualQ(filePath, float(delay_str))
    
