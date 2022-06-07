import datetime
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


def loadQAndDtForAGivenPeriod(fromDt, toDt):
    Q_all = []
    dt_all = []
    dtDiff = toDt - fromDt
    dt_crr = fromDt
    for i in range(dtDiff.days + 1):
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

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

    fetch.fetchDocsByPeriod(fromDt, toDt)

    # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)
    dt_all, Q_all = loadQAndDtForAGivenPeriod(fromDt, toDt)
    dt_all_copy = copy.deepcopy(dt_all)
    Q_all_copy = copy.deepcopy(Q_all)

    # 時系列データのデルタを均一にする
    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)

    # 時系列データの点間が全て1.0[s]かテストする
    testEqualityDeltaBetweenDts(dt_all)

    endDt = (
        dt_all[0] + datetime.timedelta(days=7) + datetime.timedelta(hours=12)
    )  # スタートから7.5日後を終わりの日時にする
    print(f"dt_all列の最大値: {endDt}")
    last_dt_index = 0
    for i, dt in enumerate(dt_all):
        if dt > endDt:
            last_dt_index = i
            break

    # 7.5日の範囲になるよう切り取る
    dt_all = dt_all[:last_dt_index]
    Q_all = Q_all[:last_dt_index]

    q_calc_end_dt = dt_all[0] + datetime.timedelta(days=7)  # スタートの日時から7日後
    q_calc_end_dt_index = 0
    for i, dt_crr in enumerate(dt_all):
        if dt_crr > q_calc_end_dt:
            q_calc_end_dt_index = i
            break
    print(f"dt_all列の先頭の日時から7日後の日時: {dt_all[q_calc_end_dt_index]}")

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    # 相互コレログラムを計算する際、計算値を6時間(0.25日)シフトさせたタイミングで計算値と実測値の時系列データのズレが消える
    dts_q_calc_all = dt_all[:q_calc_end_dt_index]
    dts_q_calc_all_with_6hours_delay = list(
        map(
            lambda dt: dt + datetime.timedelta(hours=6),
            dts_q_calc_all,
        )
    )

    print(f"dt_all[0]: {dt_all[0]}")
    print(f"dt_all[-1]: {dt_all[-1]}")
    print(f"dts_q_calc_all_with_6hours_delay[0]: {dts_q_calc_all_with_6hours_delay[0]}")
    print(
        f"dts_q_calc_all_with_6hours_delay[-1]: {dts_q_calc_all_with_6hours_delay[-1]}"
    )  # 0.25日(6時間ずらした状態からスタートして、トータル12時間シフトさせる)
    Q_calc_all = list(  # Q_allの時刻データ列より6時間遅れた時刻データ列から計算
        map(
            # lambda dt: calcQ(dt, 33.82794, 132.75093) / 1000,
            lambda dt: max(calcQ(dt, 33.82794, 132.75093), 0) / 1000,
            dts_q_calc_all_with_6hours_delay,
        )
    )

    corr = np.correlate(Q_all, Q_calc_all)

    print(f"corr.argmax(): {corr.argmax()}")
    print(
        f"dts_q_calc_all_with_6hours_delay[corr.argmax()]: {dts_q_calc_all_with_6hours_delay[corr.argmax()]}"
    )
    tmp = (
        dts_q_calc_all_with_6hours_delay[corr.argmax()]
        - dts_q_calc_all_with_6hours_delay[0]
    )
    print(f"best delta: {np.abs(6 * 60 * 60 - tmp.total_seconds())}")

    axes = [plt.subplots()[1] for i in range(2)]

    axes[0].plot(dt_all, Q_all, label="実測値(補完)")  # 実データをプロット
    axes[0].plot(dt_all_copy, Q_all_copy, label="実測値(生データ)", linestyle="dashed")
    # axes[0].set_xlim(
    #     [
    #         datetime.datetime(dt_all[0].year, dt_all[0].month, dt_all[0].day, 0, 0, 0),
    #         datetime.datetime(dt_all[0].year, dt_all[0].month, dt_all[0].day, 0, 0, 0)
    #         + datetime.timedelta(days=1),
    #     ]
    # )
    # axes[0].plot(
    #     dts_q_calc_all_with_6hours_delay,
    #     Q_calc_all,
    #     label="計算値",
    # )  # 実データをプロット
    axes[0].set_xlabel("日時")
    axes[0].set_ylabel("日射量[kW/m^2]")

    print(f"len(corr): {len(corr)}")

    axes[1].set_xlabel("ラグ")
    axes[1].set_ylabel("相互相関")
    axes[1].plot(np.arange(len(corr)), corr, color="r")

    plt.show()


if __name__ == "__main__":
    main()
