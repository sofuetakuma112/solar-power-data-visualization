import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime
from operator import itemgetter
from utils.q import calcQ
import numpy as np
from utils.es.util import sortDocsByKey, isoformats2dt, extractFieldsFromDocs
from utils.numerical_processing import min_max


def timeDiffInActualQ(filePath, delay_s):
    with open(filePath, "rb") as f:
        docs = pickle.load(f)

    docs = sortDocsByKey(docs, "JPtime")
    jptimes = extractFieldsFromDocs(docs, "JPtime")
    times = isoformats2dt(jptimes)
    qs = extractFieldsFromDocs(docs, "solarIrradiance(kw/m^2)")

    indexes = [0]
    prevDt = times[0]
    for i, time in enumerate(
        times
    ):  # 全てのdocumentをグラフ描画すると負荷がかかるので1分あたり1ドキュメントになるようデータを削る
        if prevDt.isoformat()[:16] != time.isoformat()[:16]:
            indexes.append(i)
        prevDt = time

    dtsByMinute = itemgetter(*indexes)(times)
    qs = min_max(itemgetter(*indexes)(qs))

    qs_calc = list(  # 理論値を計算(kW/m^2に変換している)
        map(
            lambda dt: max(
                calcQ(dt + datetime.timedelta(seconds=delay_s), 33.82794, 132.75093)
                / 1000,
                0,
            ),
            dtsByMinute,
        )
    )
    # qs_calc_scaled = list(map(lambda q: q * (max(qs) / max(qs_calc)), qs_calc))
    qs_calc_scaled = min_max(qs_calc)

    # 同時刻の実測値と計算値から、計算値と同じ数値を持つ実測値の時刻と、元の実測値との時刻差を求め、その最大値を計算する
    diffs_minutes = []
    for i in range(len(qs_calc_scaled)):
        # 実測値と計算値のペアから計算値を取り出して、同じ数値を取る実測値の時刻を取得する
        q_calc_scaled = qs_calc_scaled[i]

        if i < len(qs_calc_scaled) / 2:
            # 左半分で最も近い実測値を見つける
            diffs = []
            for j in range(len(qs_calc_scaled)):
                if j < len(qs_calc_scaled) / 2:
                    diffs.append(np.abs(qs[j] - q_calc_scaled))
                else:
                    diffs.append(10)
            same_q_index = diffs.index(min(diffs))
        else:
            # 右半分で最も近い実測値を見つける
            diffs = []
            for j in range(len(qs_calc_scaled)):
                if j > len(qs_calc_scaled) / 2:
                    diffs.append(np.abs(qs[j] - q_calc_scaled))
                else:
                    diffs.append(10)
            same_q_index = diffs.index(min(diffs))

        diffs_minutes.append(np.abs(i - same_q_index))

    plt.plot(
        dtsByMinute,
        diffs_minutes,
        label="計測値と同じ実測値を取る時刻と、計測値(実測値)の時刻との差",
    )

    plt.xlim(
        datetime.datetime(times[0].year, times[0].month, times[0].day, 5, 25, 0, 0),
        datetime.datetime(times[0].year, times[0].month, times[0].day, 18, 45, 0, 0),
    )

    plt.ylim(
        0,
        100,
    )

    # x 軸のラベルを設定する。
    plt.xlabel("日時")
    # y 軸のラベルを設定する。
    plt.ylabel("時間差[分]")
    plt.legend()
    plt.show()
