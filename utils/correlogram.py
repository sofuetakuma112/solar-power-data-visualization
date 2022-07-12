import datetime
from operator import itemgetter
from utils.es import fetch
from utils.file import getPickleFilePathByDatetime
from utils.date import getRelativePositionBetweenTwoDts
import pickle
from utils.es.util import sortDocsByKey, isoformats2dt, extractFieldsFromDocs
from itertools import chain
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


def loadQAndDtForPeriod(startDt, span):
    Q_all = []
    dt_all = []
    dt_crr = startDt

    span_float, span_int = math.modf(span)
    loopCount = 0
    while True:
        fetch.fetchDocsByDatetime(dt_crr)
        filePath = getPickleFilePathByDatetime(dt_crr)
        with open(filePath, "rb") as f:
            docs = pickle.load(f)
        docs = sortDocsByKey(docs, "JPtime")
        jptimes = extractFieldsFromDocs(docs, "JPtime")
        dts_per_day = isoformats2dt(jptimes)
        if loopCount == 0:
            lastDt = (
                dts_per_day[0]  # dts_per_day[0]から{fixedDaysLen}日後をlastDtとする
                + datetime.timedelta(days=span_int)
                + datetime.timedelta(hours=span_float * 24)
            )
            print(f"lastDt: {lastDt}")
        Qs_per_day = extractFieldsFromDocs(docs, "solarIrradiance(kw/m^2)")

        if all(list(map(lambda dt: dt > lastDt, dts_per_day))):
            # jptimesの全ての日時がlastDtより未来の日時
            indexes = [i for i, _ in enumerate(Q_all) if dt_all[i] <= lastDt]
            Q_all = itemgetter(*indexes)(Q_all)
            dt_all = itemgetter(*indexes)(dt_all)
            break

        Q_all = list(chain(Q_all, Qs_per_day))
        dt_all = list(chain(dt_all, dts_per_day))

        if len(Qs_per_day) < 10:
            print(f"データがない日: {dt_crr}")

        dt_crr = dt_crr + datetime.timedelta(days=1)
        loopCount += 1

    return [dt_all, Q_all]


def loadQAndDtForAGivenPeriod(fromDt, toDt):
    Q_all = []
    dt_all = []
    dtDiff = toDt - fromDt
    dt_crr = fromDt
    for i in range(dtDiff.days + 1):
        fetch.fetchDocsByDatetime(dt_crr)
        filePath = getPickleFilePathByDatetime(dt_crr)
        with open(filePath, "rb") as f:
            docs = pickle.load(f)
        docs = sortDocsByKey(docs, "JPtime")
        jptimes = extractFieldsFromDocs(docs, "JPtime")
        dts_per_day = isoformats2dt(jptimes)
        Qs_per_day = extractFieldsFromDocs(docs, "solarIrradiance(kw/m^2)")

        Q_all = list(chain(Q_all, Qs_per_day))
        dt_all = list(chain(dt_all, dts_per_day))

        if len(Qs_per_day) < 10:
            print(f"データがない日: {dt_crr}")

        dt_crr = dt_crr + datetime.timedelta(days=1)

    return [dt_all, Q_all]


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
