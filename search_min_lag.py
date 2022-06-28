import datetime
from es import fetch
import sys
from com_global import calcQ
import numpy as np
import japanize_matplotlib
import math
from correlogram import loadQAndDtForPeriod, unifyDeltasBetweenDts


def calcLag(Q_all, dts_for_calc, coef=1):
    """
    実測値と計算値の日射量データをもとに相互相関が最大となるラグ[s]を返す
    Parameters
    ----------
    Q_all : array-like
        実測値の日射量
    dts_for_calc : int
        計算値を算出する用の日時データ(スライド全量の半分だけ日時を進めている必要がある)
    coef : float
        日射量計算時に掛ける係数
    """
    Q_for_calc = list(
        map(
            lambda dt: max(calcQ(dt, 33.82794, 132.75093), 0) * coef / 1000,
            dts_for_calc,
        )
    )

    corr = np.correlate(Q_all, Q_for_calc)

    largest_lag_sec = 6 * 60 * 60 - corr.argmax()  # 真の計算値の時間 - 実測値の時間
    return largest_lag_sec


def shiftDts(dts, days, hour_coef):
    """
    指定した期間だけ日時データを進める(遅らせる)
    Parameters
    ----------
    dts : array-like
        日付データ
    days : int
        進める(遅らせる)日数
    hour_coef : float
        24時間のうち、何時間進める(遅らせる)かの係数
    """
    return list(
        map(
            lambda dt: dt
            + datetime.timedelta(days=days)
            + datetime.timedelta(hours=hour_coef * 24),
            dts,
        )
    )


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
    dts_for_calc = dt_all[:q_calc_end_dt_index]

    # TODO: 以下ループ
    # 相互相関を求めるために計算値の日時を(スライド全量 / 2)だけずらす
    dts_for_calc_applied_lag = shiftDts(dts_for_calc, dtStartLag_int, dtStartLag_float)

    # 相互相関が最大となるラグを返す
    largest_lag_sec = calcLag(Q_all, dts_for_calc_applied_lag)
    print(f"largest_lag_sec: {largest_lag_sec}")
    dts_applied_largest_lag_sec = list(
        map(
            lambda dt: dt + datetime.timedelta(seconds=int(largest_lag_sec)),
            dts_for_calc,
        )
    )

    dts_for_calc_applied_lag_and_half_slides = shiftDts(
        dts_applied_largest_lag_sec, dtStartLag_int, dtStartLag_float
    )

    lags_sec = []
    for coef in list(
        map(lambda x: x / 10, range(1, 11))
    ):  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        largest_lag_sec = calcLag(Q_all, dts_for_calc_applied_lag_and_half_slides, coef)
        print(f"coef: {coef}, largest_lag_sec: {largest_lag_sec}")
        lags_sec.append(largest_lag_sec)

    print(f"min(lags_sec): {min(lags_sec)}")


if __name__ == "__main__":
    main()
