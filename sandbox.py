import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime
from operator import itemgetter

from com_global import calcQ
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

    A = [1, 2, 3]
    B = [0, 1, 0.5]
    A = np.array(A)
    B = np.array(B)
    cov = np.cov(A, B)[0][1]
    A_std = np.std(A)
    B_std = np.std(B)
    print(cov / (A_std * B_std))
    print(np.correlate(A, B))
