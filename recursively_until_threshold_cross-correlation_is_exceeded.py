import datetime
from utils.es.fetch import fetchDocsByPeriod
import sys
from utils.q import calcQ
import numpy as np
import japanize_matplotlib
import math
from utils.correlogram import (
    testEqualityDeltaBetweenDts,
    unifyDeltasBetweenDts,
    calcQCalcEndDtIndex,
    slidesQCalcForCorr,
)
from utils.es.load import loadQAndDtForPeriod
import csv


# > python3 recursively_until_threshold_cross-correlation_is_exceeded.py 2022/04/01 2.5 2 0.27
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
        with open('data/csv/corr_avg_lag.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([fromDt, fixedDaysLen, dynamicDaysLen, corr_max, lag])
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
