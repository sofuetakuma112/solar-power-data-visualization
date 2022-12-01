import datetime
import sys
from utils.q import calc_q_kw
import numpy as np
import japanize_matplotlib
import math
import matplotlib.pyplot as plt
from utils.correlogram import shiftDts, testEqualityDeltaBetweenDts


def main():
    args = sys.argv

    fromDtStr = args[1].split("/")
    fixedDaysLen = float(args[2])
    dynamicDaysLen = float(args[3])
    start = int(args[4])
    last = int(args[5])
    coef = float(args[6])

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))

    # TODO: dtsをdelta t = 1s で作る
    dts = [fromDt]
    fixed_float, fixed_int = math.modf(fixedDaysLen)
    toDt = (
        fromDt
        + datetime.timedelta(days=fixed_int)
        + datetime.timedelta(hours=fixed_float * 24)
    )
    print(f"toDt: {toDt}")

    while True:
        newDt = dts[-1] + datetime.timedelta(seconds=1 * coef)
        if newDt > toDt:
            break
        else:
            dts.append(newDt)

    testEqualityDeltaBetweenDts(dts, 1 * coef)

    Q_all = list(
        map(
            calc_q_kw,
            dts,
        )
    )

    # 実測値の日時データからトリムして計算値用の日時データを作るので
    # トリムする範囲を指定するためのインデックスを求める
    q_calc_end_dt = dts[0] + datetime.timedelta(days=dynamicDaysLen)
    print(f"len(dts): {len(dts)}")
    print(f"q_calc_end_dt: {q_calc_end_dt}")
    q_calc_end_dt_index = 0
    for i, dt_crr in enumerate(dts):
        if dt_crr > q_calc_end_dt:
            print(f"break with index: {i}")
            q_calc_end_dt_index = i
            break

    print(f"q_calc_end_dt_index: {q_calc_end_dt_index}")

    dts_for_calc = dts[:q_calc_end_dt_index]

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    dtStartLag_float, dtStartLag_int = math.modf((fixedDaysLen - dynamicDaysLen) / 2)
    dts_for_calc_applied_lag = shiftDts(dts_for_calc, dtStartLag_int, dtStartLag_float)

    Q_for_calc = list(
        map(
            calc_q_kw,
            dts_for_calc_applied_lag,
        )
    )

    print(f"len(Q_all): {len(Q_all)}")
    print(f"len(Q_for_calc): {len(Q_for_calc)}")
    corr = np.correlate(Q_all, Q_for_calc)
    largest_lag_sec = 6 * 60 * 60 - corr.argmax()  # 真の計算値の時間 - 実測値の時間
    print(f"largest_lag_sec: {largest_lag_sec}")

    diffs = [
        i - (fixedDaysLen - dynamicDaysLen) * 24 * 60 * 60 / 2 for i in range(len(corr))
    ]

    corrWithDiff_list = []
    for i, diff in enumerate(diffs):
        corrWithDiff_list.append([diff, corr[i]])

    filteredCorrWithDiff_list = list(filter(
        lambda corrWithDiff: corrWithDiff[0] >= start and corrWithDiff[0] <= last,
        corrWithDiff_list,
    ))

    axes = [plt.subplots() for _ in range(1)]
    axes[0][1].plot(
        list(map(lambda l: l[0], filteredCorrWithDiff_list)),
        list(map(lambda l: l[1], filteredCorrWithDiff_list)),
        color="r",
    )
    axes[0][1].set_xlabel("2つの時系列データの日時のズレ[s]")
    axes[0][1].set_ylabel("相互相関")

    plt.show()


if __name__ == "__main__":
    main()
