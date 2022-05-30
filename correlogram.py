import datetime
from es import fetch
from utils.file import getPickleFilePathByDatetime
from utils.numerical_processing import min_max
import sys
import pickle
from es.util import sortDocsByKey, isoformats2dt, extractFieldsFromDocs
from itertools import chain
from com_global import calcQ
import matplotlib.pyplot as plt
import numpy as np

args = sys.argv
fromDtStr = args[1].split("/")
toDtStr = args[2].split("/")

fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

fetch.fetchDocsByPeriod(fromDt, toDt)

Q_all = []
dt_all = []
# 指定した範囲の実測データを全て時系列順で結合する
dtDiff = toDt - fromDt
for i in range(dtDiff.days + 1):
    dt_crr = fromDt
    filePath = getPickleFilePathByDatetime(dt_crr)
    with open(filePath, "rb") as f:
        docs = pickle.load(f)
    docs = sortDocsByKey(docs, "JPtime")
    jptimes = extractFieldsFromDocs(docs, "JPtime")
    dts_per_day = isoformats2dt(jptimes)
    Qs_per_day = extractFieldsFromDocs(docs, "solarIrradiance(kw/m^2)")

    Q_all = list(chain(Q_all, Qs_per_day))
    dt_all = list(chain(dt_all, dts_per_day))

    # print(f"len(Qs_per_day): {len(Qs_per_day)}")
    # print(f"len(dts_per_day): {len(dts_per_day)}")
    # print(f"len(Q_all): {len(Q_all)}")
    # print(f"len(dt_all): {len(dt_all)}")

Q_all = min_max(Q_all)
Q_calc_all = min_max(
    list(map(lambda dt: calcQ(dt, 33.82794, 132.75093) / 1000, dt_all))
)

corr = np.correlate(Q_all, Q_calc_all, "full")
# corr = np.correlate(Q_all, Q_all, "full")
delay = corr.argmax() - (len(Q_all) - 1)
print(str(delay))
print(corr.max())

plt.subplot(1, 1, 1)
plt.ylabel("corr")
plt.plot(np.arange(len(corr)) - len(Q_all) + 1, corr, color="r")
plt.xlim([0, len(Q_all)])

plt.show()
