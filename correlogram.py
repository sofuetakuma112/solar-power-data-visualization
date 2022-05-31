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
import japanize_matplotlib

args = sys.argv
fromDtStr = args[1].split("/")
toDtStr = args[2].split("/")

fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

fetch.fetchDocsByPeriod(fromDt, toDt)

Q_all = []
dt_all = []

dt_lengths = []
# 指定した範囲の実測データを全て時系列順で結合する
dtDiff = toDt - fromDt
dt_crr = fromDt
for i in range(dtDiff.days + 1):
    # print(i, dt_crr)
    filePath = getPickleFilePathByDatetime(dt_crr)
    with open(filePath, "rb") as f:
        docs = pickle.load(f)
    docs = sortDocsByKey(docs, "JPtime")
    jptimes = extractFieldsFromDocs(docs, "JPtime")
    dts_per_day = isoformats2dt(jptimes)
    Qs_per_day = extractFieldsFromDocs(docs, "solarIrradiance(kw/m^2)")

    dt_lengths.append(len(dts_per_day))

    Q_all = list(chain(Q_all, Qs_per_day))
    dt_all = list(chain(dt_all, dts_per_day))

    if len(Qs_per_day) < 10:
        print(f"データがない日: {dt_crr}")

    dt_crr = dt_crr + datetime.timedelta(days=1)

    # print(f"len(Qs_per_day): {len(Qs_per_day)}")
    # print(f"len(dts_per_day): {len(dts_per_day)}")
    # print(f"len(Q_all): {len(Q_all)}")
    # print(f"len(dt_all): {len(dt_all)}")

Q_all = min_max(Q_all)
Q_calc_all = min_max(
    list(map(lambda dt: calcQ(dt, 33.82794, 132.75093) / 1000, dt_all))
)
# Q_calc_all = list(map(lambda dt: calcQ(dt, 33.82794, 132.75093) / 1000, dt_all))

# corr = np.correlate(Q_all, Q_all, "full")
# print(f"dt_lengths: {dt_lengths}")
# print(f"sum(dt_lengths[:3]): {sum(dt_lengths[:3])}")
print(
    f"len(Q_all) - len(Q_calc_all[:sum(dt_lengths[:3])]): {len(Q_all) - len(Q_calc_all[:sum(dt_lengths[:3])])}"
)
corr = np.correlate(Q_all, Q_calc_all[: sum(dt_lengths[:3])])
print(f"corr.argmax(): {corr.argmax()}")
delay = corr.argmax() - (len(Q_all) - 1)
print(f"str(delay): {str(delay)}")
print(f"corr.max(): {corr.max()}")

# print(f"len(dt_all): {len(dt_all)}")
print(f"len(Q_all): {len(Q_all)}")
print(f"len(Q_calc_all[:sum(dt_lengths[:3])]): {len(Q_calc_all[:sum(dt_lengths[:3])])}")
# print(f"dt_all[0], dt_all[-1]", dt_all[0], dt_all[-1])
# plt.plot(dt_all, Q_all, label="実測値")  # 実データをプロット

print(f"len(corr): {len(corr)}")

plt.subplot(1, 1, 1)
plt.ylabel("corr")
plt.plot(np.arange(len(corr)), corr, color="r")
# plt.xlim([0, len(Q_all)])

# x 軸のラベルを設定する。
# plt.xlabel("日時")
# y 軸のラベルを設定する。
plt.ylabel("日射量[kW/m^2]")
plt.legend()
plt.show()
