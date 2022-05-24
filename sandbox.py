import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime
from operator import itemgetter
from com_global import calcQ
import sys
import os


def plotQ():
    for i in range(11):
        # datetimeの分刻みのリストを作成する
        dtsByMinute = list(
            map(
                lambda j: datetime.datetime(2022, i + 1, 1, 0, 0, 0) + datetime.timedelta(minutes=j),
                range(1440),
            )
        )
        qs_theoretical = list(  # 理論値を計算(kW/m^2に変換している)
            map(lambda dt: max(calcQ(dt, 33.82794, 132.75093) / 1000, 0), dtsByMinute)
        )
        plt.plot(  # 理論値をプロット
            list(map(lambda i: datetime.datetime(2022, 1, 1, 0, 0, 0) + datetime.timedelta(minutes=i), range(1440))),
            list(qs_theoretical),  # 縦軸のスケールを実測値と揃えている
            label=f"理論値({i + 1}月)",
        )

        # x 軸のラベルを設定する。
        plt.xlabel("日時")

        # y 軸のラベルを設定する。
        plt.ylabel("日射量[kW/m^2]")
        plt.legend()

    plt.show()


if __name__ == "__main__":
    plotQ()
