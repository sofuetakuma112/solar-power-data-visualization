import datetime
from utils.es.fetch import fetchDocsByPeriod
import sys
from utils.q import calcQ
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import copy
import math
from utils.correlogram import (
    testEqualityDeltaBetweenDts,
    unifyDeltasBetweenDts,
)
from utils.es.load import loadQAndDtForPeriod

# 実測値の日時データからトリムして計算値用の日時データを作るので
# トリムする範囲を指定するためのインデックスを求める
def calcQCalcEndDtIndex(dts, dynamicSpanLen):
    q_calc_end_dt = dts[0] + datetime.timedelta(days=dynamicSpanLen)
    for i, dt_crr in enumerate(dts):
        if dt_crr > q_calc_end_dt:
            return i
    raise ValueError("実測値の日時列の長さが足りない")


# 相互相関の計算のために、計算値の日射量データの時系列データをずらす
def slidesQCalcForCorr(dts, dt_last_index, fixedSpanLen, dynamicSpanLen):
    dtStartLag_float, dtStartLag_int = math.modf((fixedSpanLen - dynamicSpanLen) / 2)

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    # 相互コレログラムを計算する際、計算値を{(fixedSpanLen - dynamicSpanLen) * 24 / 2}時間({(fixedSpanLen - dynamicSpanLen) / 2}日)シフトさせたタイミングで計算値と実測値の時系列データのズレが消える
    dts_q_calc_all = dts[:dt_last_index]
    dts_q_calc_all_with_lag = list(
        map(
            lambda dt: dt
            + datetime.timedelta(days=dtStartLag_int)
            + datetime.timedelta(hours=dtStartLag_float * 24),
            dts_q_calc_all,
        )
    )

    return list(
        map(
            lambda dt: max(calcQ(dt, 33.82794, 132.75093), 0) / 1000,
            dts_q_calc_all_with_lag,
        )
    )


# 相互相関の平均の最大値とその時のラグを返す
def calcCorr(fromDt, toDt, fixedSpanLen, dynamicSpanLen):
    fetchDocsByPeriod(fromDt, toDt)  # pickleファイルがない場合、取得する
    dt_all, Q_all = loadQAndDtForPeriod(
        fromDt, fixedSpanLen
    )  # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)
    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)  # 時系列データのデルタを均一にする
    testEqualityDeltaBetweenDts(dt_all)  # 時系列データの点間が全て1.0[s]かテストする

    q_calc_end_dt_index = calcQCalcEndDtIndex(dt_all, dynamicSpanLen)

    # Q_calc_allの時系列データを実測値の時系列データよりx時間進める
    Q_calc_all_applied_lag = slidesQCalcForCorr(
        dt_all, q_calc_end_dt_index, fixedSpanLen, dynamicSpanLen
    )

    corr = np.correlate(Q_all, Q_calc_all_applied_lag)

    largest_lag_sec = 6 * 60 * 60 - corr.argmax()
    print(f"真の計算値の時間 - 実測値の時間: {largest_lag_sec}")

    return corr.max() / len(Q_calc_all_applied_lag), largest_lag_sec


# 相互相関の平均値の最大が指定したしきい値を超えるまで再帰的に相互相関を求める
def main():
    args = sys.argv

    fromDtStr = args[1].split("/")
    fixedDaysLen = float(args[2])
    dynamicDaysLen = float(args[3])
    threshold = float(args[4])

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = fromDt + datetime.timedelta(days=math.ceil(fixedDaysLen))

    while True:
        corr_max, lag = calcCorr(fromDt, toDt, fixedDaysLen, dynamicDaysLen)
        print(f"corr_max: {corr_max}")
        if corr_max > threshold:  # しきい値を超えた
            break
        # スパンの更新
        fixedDaysLen += 1
        dynamicDaysLen += 1
        toDt = fromDt + datetime.timedelta(days=math.ceil(fixedDaysLen))

    print(
        f"結果: fromDt: {fromDt}, fixedDaysLen: {fixedDaysLen}, dynamicDaysLen: {dynamicDaysLen}, lag: {lag}, 相互相関の平均の最大値: {corr_max}"
    )


if __name__ == "__main__":
    main()
