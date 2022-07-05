from com_global import calcQ
import numpy as np
import datetime

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

def testEqualityDeltaBetweenDts(dts, delta=1.0):
    """
    日時データのdeltaが1[s]で統一されているかテストする
    """
    for i in range(len(dts)):
        if len(dts) - 1 == i:
            break
        diff = dts[i + 1] - dts[i]
        if diff.total_seconds() != delta:  # 日時のデルタが1sではない
            print(f"i: {i}")
            print(f"dts[i + 1]: {dts[i + 1]}")
            print(f"dts[i]: {dts[i]}")
            print(diff.total_seconds())
            raise ValueError("error!")