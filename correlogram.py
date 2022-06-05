import datetime
from es import fetch
from utils.file import getPickleFilePathByDatetime
from utils.numerical_processing import min_max
import sys
import pickle
from es.util import sortDocsByKey, isoformats2dt, extractFieldsFromDocs
from itertools import chain
from com_global import calcQ
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib


def main():
    args = sys.argv

    fromDtStr = args[1].split("/")
    toDtStr = args[2].split("/")

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

    fetch.fetchDocsByPeriod(fromDt, toDt)

    Q_all = []
    dt_all = []

    dt_lengths = []
    # 指定した範囲の実測データを全て時系列順で結合する
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

        dt_lengths.append(len(dts_per_day))

        Q_all = list(chain(Q_all, Qs_per_day))
        dt_all = list(chain(dt_all, dts_per_day))

        if len(Qs_per_day) < 10:
            print(f"データがない日: {dt_crr}")

        dt_crr = dt_crr + datetime.timedelta(days=1)

    # TODO: 時刻データ列のdeltaを均一にする
    # print(f"dt_all[:100]: {dt_all[:100]}")
    # # dtのミリ秒を全て0に統一する
    # dt_all = list(
    #     map(
    #         lambda dt: datetime.datetime(
    #             dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0
    #         ),
    #         dt_all,
    #     )
    # )
    # # dt_allのdeltaを1秒に統一する
    # dt_prev = dt_all[0]
    # for i, dt_crr in enumerate(dt_all):
    #     diff = dt_crr - dt_prev
    #     diff_seconds = diff.total_seconds()
    #     if dt_prev.isoformat()[:19] == dt_crr.isoformat()[:19]:
    #         dt_prev = dt_crr
    #         continue
    #     elif diff_seconds >= 2:
    #         print(f"delta: {diff_seconds}, between: {dt_crr}, {dt_prev}")
    #     # else:
    #     #     # 正常
    #     dt_prev = dt_crr

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

    # plt.subplot(1, 2, 1)
    # plt.plot(dt_all, Q_all, label="実測値")  # 実データをプロット
    # plt.plot(
    #     dts_q_calc_all_with_6hours_delay,
    #     Q_calc_all,
    #     label="計算値",
    # )  # 実データをプロット

    print(f"len(corr): {len(corr)}")

    plt.subplot(1, 1, 1)
    plt.xlabel("ラグ")
    plt.ylabel("相互相関")
    plt.plot(np.arange(len(corr)), corr, color="r")
    # plt.xlim([0, len(Q_all)])

    # x 軸のラベルを設定する。
    # plt.xlabel("日時")
    # y 軸のラベルを設定する。
    plt.show()


if __name__ == "__main__":
    main()
