import datetime
from utils.date import getRelativePositionBetweenTwoDts
from utils.q import calc_q_kw
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
            lambda dt: calc_q_kw(dt) * coef,
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


class NotEnoughLengthErr:
    def __init__(self):
        self.message = "実測値の日時列の長さが足りない"


def calc_dts_for_q_calc(dts, dynamicSpanLen):
    """
    実測値の日時データからトリムして計算値用の日時データを作るので
    トリムする範囲を指定するためのインデックスを求める
    """
    q_calc_end_dt = dts[0] + datetime.timedelta(days=dynamicSpanLen)
    mask = dts <= q_calc_end_dt
    if np.all(mask):
        return NotEnoughLengthErr()
    else:
        return dts[mask]

    # dts[mask]
    # end_idx = -1
    # for i, dt_crr in enumerate(dts):
    #     if dt_crr > q_calc_end_dt:
    #         end_idx = i
    #         return dts[:end_idx]
    # return NotEnoughLengthErr()


# 相互相関の計算のために、計算値の日射量データの時系列データをずらす
def slides_q_calc_for_corr(dts_for_q_calc, fixedSpanLen, dynamicSpanLen):
    dtStartLag_float, dtStartLag_int = math.modf((fixedSpanLen - dynamicSpanLen) / 2)

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    # 相互コレログラムを計算する際、計算値を{(fixedSpanLen - dynamicSpanLen) * 24 / 2}時間({(fixedSpanLen - dynamicSpanLen) / 2}日)シフトさせたタイミングで計算値と実測値の時系列データのズレが消える
    # dts_for_q_calc_with_lag = list(
    #     map(
    # lambda dt: dt
    # + datetime.timedelta(days=dtStartLag_int)
    # + datetime.timedelta(hours=dtStartLag_float * 24),
    #         dts_for_q_calc,
    #     )
    # )

    dts_for_q_calc_with_lag = np.vectorize(
        lambda dt: dt
        + datetime.timedelta(days=dtStartLag_int)
        + datetime.timedelta(hours=dtStartLag_float * 24)
    )(dts_for_q_calc)

    # return list(
    #     map(
    #         calc_q_kw,
    #         dts_for_q_calc_with_lag,
    #     )
    # )

    return np.vectorize(calc_q_kw)(dts_for_q_calc_with_lag)


def calc_ratios(dts, qs):
    """
    実測値と同じ日時の計算値を求めて、実測値と計算値の比を返す
    """
    # Qs_calc = list(
    #     map(
    #         calc_q_kw,
    #         dts,
    #     )
    # )
    Qs_calc = np.vectorize(calc_q_kw)(dts)

    def calc_ratio(l):
        q, q_calc = l
        ratio = None
        if q_calc < 0.001:  # 計算値が0の部分は比を1扱いにする
            ratio = 1
        else:
            ratio = q / q_calc
        return ratio

    q_and_q_calc_ndarray = np.concatenate(
        [qs.reshape([-1, 1]), Qs_calc.reshape([-1, 1])], 1
    )

    ratios = np.apply_along_axis(calc_ratio, 1, q_and_q_calc_ndarray)

    # for q, q_calc in zip(
    #     qs,
    #     Qs_calc,
    # ):
    #     ratio = None
    #     if q_calc < 0.001:  # 計算値が0の部分は比を1扱いにする
    #         ratio = 1
    #     else:
    #         ratio = q / q_calc
    #     ratios.append(ratio)

    return ratios
