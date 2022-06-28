import datetime
from es import fetch
import sys
from com_global import calcQ
import numpy as np
import japanize_matplotlib
import math
from correlogram import loadQAndDtForPeriod, unifyDeltasBetweenDts


def calcLag(Q_all, dts_q_calc_all_applied_lag, coef=1):
    Q_calc_all_applied_lag = list(
        map(
            lambda dt: max(calcQ(dt, 33.82794, 132.75093), 0) * coef / 1000,
            dts_q_calc_all_applied_lag,
        )
    )

    corr = np.correlate(Q_all, Q_calc_all_applied_lag)

    largest_lag_sec = 6 * 60 * 60 - corr.argmax()  # 真の計算値の時間 - 実測値の時間
    return largest_lag_sec


def main():
    args = sys.argv

    fromDtStr = args[1].split("/")
    toDtStr = args[2].split("/")
    fixedDaysLen = float(args[3])
    dynamicDaysLen = float(args[4])

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

    fetch.fetchDocsByPeriod(fromDt, toDt)  # pickleファイルがない場合、取得する

    # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)
    dt_all, Q_all = loadQAndDtForPeriod(fromDt, fixedDaysLen)
    # 時系列データのデルタを均一にする
    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)

    # 実測値の日時データからトリムして計算値用の日時データを作るので
    # トリムする範囲を指定するためのインデックスを求める
    q_calc_end_dt = dt_all[0] + datetime.timedelta(days=dynamicDaysLen)
    q_calc_end_dt_index = 0
    for i, dt_crr in enumerate(dt_all):
        if dt_crr > q_calc_end_dt:
            q_calc_end_dt_index = i
            break

    dtStartLag_float, dtStartLag_int = math.modf((fixedDaysLen - dynamicDaysLen) / 2)

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    # 相互コレログラムを計算する際、計算値を{(fixedDaysLen - dynamicDaysLen) * 24 / 2}時間({(fixedDaysLen - dynamicDaysLen) / 2}日)シフトさせたタイミングで計算値と実測値の時系列データのズレが消える
    dts_q_calc_all = dt_all[:q_calc_end_dt_index]
    dts_q_calc_all_applied_lag = list(
        map(
            lambda dt: dt
            + datetime.timedelta(days=dtStartLag_int)
            + datetime.timedelta(hours=dtStartLag_float * 24),
            dts_q_calc_all,
        )
    )

    # 以下ループ

    # 相互相関が最大となるラグを返す
    largest_lag_sec = calcLag(Q_all, dts_q_calc_all_applied_lag)
    print(f"largest_lag_sec: {largest_lag_sec}")
    dts_applied_largest_lag_sec = list(
        map(
            lambda dt: dt + datetime.timedelta(seconds=int(largest_lag_sec)),
            dts_q_calc_all,
        )
    )

    lags_sec = []
    for coef in list(
        map(lambda x: x / 10, range(1, 11))
    ):  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        largest_lag_sec = calcLag(Q_all, dts_applied_largest_lag_sec, coef) # TODO: これを更に6時間ずらして相互相関を計算する必要がある
        print(f"coef: {coef}")
        print(f"largest_lag_sec: {largest_lag_sec}")
        lags_sec.append(largest_lag_sec)

    print(f"min(lags_sec): {min(lags_sec)}")


if __name__ == "__main__":
    main()
