import datetime
from utils.date import getRelativePositionBetweenTwoDts
from utils.q import calcQ
import numpy as np
import math

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

# データがない日が含めて読み込んだ場合、エラーを投げるようにしたのでこの関数は不要
# def checkAnyDaysWithNoDataIncluded(startDt, span):
#     dt_all, Q_all = loadQAndDtForPeriod(startDt, span)
    
#     # 日毎にスプリットしてデータをチェックする
#     baseDt = dt_all[0]
#     crrIdx = 1
#     qs_per_day = []
#     while True:
#         crrDt = dt_all[crrIdx]
#         if baseDt.year == crrDt.year and baseDt.month == crrDt.month and baseDt.day == crrDt.day:
#             # まだ同じ日付
#             qs_per_day.append(Q_all[crrIdx])
#         else:
#             # 次の日付に移った
#             baseDt = dt_all[crrIdx]
            
#             # 日射量データ列が有効かチェック
            
            
#             # 日射量データ列を初期化
#             qs_per_day = []
#         crrIdx += 1
#         if crrIdx > len(dt_all):
#             break


def getDtListAndCoefBeCompleted(dts, qs):
    """
    補完した日時データと日射量の増分に対する係数のリストを取得する
    """
    comps = []
    for i in range(len(dts)):
        if len(dts) - 1 == i:  # 最後の要素を参照するインデックスの場合、次の要素の参照時にエラーが起こるのでスキップ
            break
        dt_crr = dts[i]
        dt_next = dts[i + 1]
        q_crr = qs[i]
        q_next = qs[i + 1]
        q_delta = q_next - q_crr

        # 補完する時刻と⊿yの係数のリストを取得する
        dt_and_increment_coef_list = getRelativePositionBetweenTwoDts(dt_crr, dt_next)
        for dt_and_coef in dt_and_increment_coef_list:
            dt_comp = dt_and_coef[0]

            increment_coef = dt_and_coef[1]
            q_comp = q_crr + q_delta * increment_coef

            comps.append([dt_comp, q_comp])
    return comps


def unifyDeltasBetweenDts(dts, qs):
    """
    補完した日時データとそれに対応した日射量のリストを取得する
    """
    # 補完データを取得する
    comps = getDtListAndCoefBeCompleted(dts, qs)
    # 補完データと既存のデータをマージする
    dt_and_q_list = []
    for i in range(len(qs)):
        dt_and_q_list.append([dts[i], qs[i]])
    merged_dt_and_q_list = dt_and_q_list + comps
    # dtの昇順でソートする
    merged_dt_and_q_list = sorted(  # datetimeでソート
        merged_dt_and_q_list,
        key=lambda dt_and_q: dt_and_q[0],
    )
    # datetimeのミリ秒が0でないデータを除外する
    merged_dt_and_q_list = list(
        filter(lambda dt_and_q: dt_and_q[0].microsecond == 0, merged_dt_and_q_list)
    )
    return [
        list(map(lambda dt_and_q: dt_and_q[0], merged_dt_and_q_list)),
        list(map(lambda dt_and_q: dt_and_q[1], merged_dt_and_q_list)),
    ]

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