import datetime
from operator import itemgetter
from es import fetch
from utils.file import getPickleFilePathByDatetime
from utils.numerical_processing import min_max
from utils.date import getRelativePositionBetweenTwoDts
import sys
import pickle
from es.util import sortDocsByKey, isoformats2dt, extractFieldsFromDocs
from itertools import chain
from com_global import calcQ
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import copy
import math


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
    dt_all_copy = copy.deepcopy(dt_all)  # 補完が正しく行えているか確認する用
    Q_all_copy = copy.deepcopy(Q_all)  # 補完が正しく行えているか確認する用

    print(f"dt_all[0]: {dt_all[0]}")
    print(f"dt_all[-1]: {dt_all[-1]}")

    # 時系列データのデルタを均一にする
    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)

    # 時系列データの点間が全て1.0[s]かテストする
    testEqualityDeltaBetweenDts(dt_all)

    # 実測値の日時データからトリムして計算値用の日時データを作るので
    # トリムする範囲を指定するためのインデックスを求める
    q_calc_end_dt = dt_all[0] + datetime.timedelta(days=dynamicDaysLen)
    q_calc_end_dt_index = 0
    for i, dt_crr in enumerate(dt_all):
        if dt_crr > q_calc_end_dt:
            q_calc_end_dt_index = i
            break
    print(f"dt_all列の先頭の日時から{dynamicDaysLen}日後の日時: {dt_all[q_calc_end_dt_index]}")

    dtStartLag_float, dtStartLag_int = math.modf((fixedDaysLen - dynamicDaysLen) / 2)

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    # 相互コレログラムを計算する際、計算値を{(fixedDaysLen - dynamicDaysLen) * 24 / 2}時間({(fixedDaysLen - dynamicDaysLen) / 2}日)シフトさせたタイミングで計算値と実測値の時系列データのズレが消える
    dts_q_calc_all = dt_all[:q_calc_end_dt_index]
    dts_q_calc_all_with_lag = list(
        map(
            lambda dt: dt
            + datetime.timedelta(days=dtStartLag_int)
            + datetime.timedelta(hours=dtStartLag_float * 24),
            dts_q_calc_all,
        )
    )

    print(
        f"dts_q_calc_all_with_{(fixedDaysLen - dynamicDaysLen) * 24 / 2}hours_delay[0]: {dts_q_calc_all_with_lag[0]}"
    )
    print(
        f"dts_q_calc_all_with_{(fixedDaysLen - dynamicDaysLen) * 24 / 2}hours_delay[-1]: {dts_q_calc_all_with_lag[-1]}"
    )
    Q_calc_all_applied_lag = list(
        map(
            lambda dt: max(calcQ(dt, 33.82794, 132.75093), 0) / 1000,
            dts_q_calc_all_with_lag,
        )
    )

    corr = np.correlate(Q_all, Q_calc_all_applied_lag)

    print(f"{corr.argmax()}秒スライドさせたとき相互相関が最大")  # corr.argmax()秒スライドさせた時が相互相関が最大
    largest_lag_sec = 6 * 60 * 60 - corr.argmax()
    print(f"真の計算値の時間 - 実測値の時間: {largest_lag_sec}")

    # axes = [plt.subplots()[1] for i in range(2)]
    axes = [plt.subplots() for _ in range(2)]

    # axes[0].plot(dt_all, Q_all, label="実測値(補完)")  # 補完データをプロット
    axes[0][1].plot(dt_all_copy, Q_all_copy, label="実測値", linestyle="dashed")

    print(int(largest_lag_sec))
    slided_dts_with_largest_lag_sec = list(
        map(
            lambda dt: dt + datetime.timedelta(seconds=int(largest_lag_sec)),
            dts_q_calc_all,
        )
    )
    axes[0][1].plot(
        dts_q_calc_all,
        list(
            map(
                lambda dt: max(calcQ(dt, 33.82794, 132.75093), 0) / 1000,
                slided_dts_with_largest_lag_sec,
            )
        ),
        label="計算値(相互相関が最大となるラグを適用)",
        linestyle="dashed",
    )

    axes[0][1].set_xlabel("日時")
    axes[0][1].set_ylabel("日射量[kW/m^2]")

    print(f"len(corr): {len(corr)}")

    axes[1][1].set_xlabel("実測値の日時 - 計算値の日時[s]")
    axes[1][1].set_ylabel("相互相関")
    axes[1][1].plot(
        [
            i - (fixedDaysLen - dynamicDaysLen) * 24 * 60 * 60 / 2
            for i in range(len(corr))
        ],
        corr,
        color="r",
    )

    axes[0][0].legend()
    plt.show()


if __name__ == "__main__":
    main()
