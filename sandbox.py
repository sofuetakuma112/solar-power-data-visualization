import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime
from operator import itemgetter
from utils.es.fetch import fetchDocsByPeriod

from utils.q import calcQ
import sys
import os

import numpy as np
from itertools import chain


# def plotQ():
#     for i in range(11):
#         # datetimeの分刻みのリストを作成する
#         dtsByMinute = list(
#             map(
#                 lambda j: datetime.datetime(2022, i + 1, 1, 0, 0, 0) + datetime.timedelta(minutes=j),
#                 range(1440),
#             )
#         )
#         qs_theoretical = list(  # 理論値を計算(kW/m^2に変換している)
#             map(lambda dt: max(calcQ(dt, 33.82794, 132.75093) / 1000, 0), dtsByMinute)
#         )
#         plt.plot(  # 理論値をプロット
#             list(map(lambda i: datetime.datetime(2022, 1, 1, 0, 0, 0) + datetime.timedelta(minutes=i), range(1440))),
#             list(qs_theoretical),  # 縦軸のスケールを実測値と揃えている
#             label=f"理論値({i + 1}月)",
#         )

#         # x 軸のラベルを設定する。
#         plt.xlabel("日時")

#         # y 軸のラベルを設定する。
#         plt.ylabel("日射量[kW/m^2]")
#         plt.legend()

#     plt.show()


if __name__ == "__main__":
    # plotQ()
    # nums1 = np.array([1, 2, 3])
    # nums2 = np.array([1] * len(nums1))

    # print((nums1 - nums2) / 5)
    # print((nums1 - 1) / 5)

    # A = []
    # for i in range(5):
    #     B = [i] * 3
    #     B = [i] * 3
    #     A = list(chain(A, B))

    # print(A)

    # A = [1, 2, 3]
    # B = [0, 1, 0.5]
    # A = np.array(A)
    # B = np.array(B)
    # cov = np.cov(A, B)[0][1]
    # A_std = np.std(A)
    # B_std = np.std(B)
    # print(cov / (A_std * B_std))
    # print(np.correlate(A, B))

    # dt1 = datetime.datetime(2018, 12, 31, 5, 0, 30, 1000)
    # dt2 = datetime.datetime(2018, 12, 31, 5, 0, 31, 500)
    # dt3 = datetime.datetime(2018, 12, 31, 5, 0, 33, 500)
    # # print(dt.isoformat()[:19])

    # diff1 = dt2 - dt1
    # diff2 = dt3 - dt2
    # print(diff1.total_seconds() * 1000)
    # print(diff2.total_seconds() * 1000)

    # list = [1, 2, 3, 4, 5]
    # for i in range(len(list)):
    #     print(i)

    # dt5 = datetime.datetime(2018, 12, 31, 5, 1, 11, 300000)
    # dt6 = datetime.datetime(2018, 12, 31, 5, 1, 12)
    # dt7 = datetime.datetime(2018, 12, 31, 5, 1, 12)
    # dt8 = datetime.datetime(2018, 12, 31, 5, 1, 14, 200000)

    # print(dt5.microsecond)
    # print(dt6.microsecond)
    # print(dt7.microsecond)
    # print(dt8.microsecond)

    # print(datetime.datetime(2022, 5, 21, 0, 4, 7, 200515).microsecond == 0)

    # print(list(map(lambda x: x / 10, range(1, 11))))

    # print(calcQ(datetime.datetime(2022, 4, 1, 12, 0, 0), 33.82794, 132.75093))
    # print(calcQ(datetime.datetime(2022, 4, 1, 12, 0, 1), 33.82794, 132.75093))

    fetchDocsByPeriod(datetime.datetime(2022, 1, 1), datetime.datetime(2022, 10, 20))
