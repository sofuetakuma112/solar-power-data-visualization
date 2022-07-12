import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime
from operator import itemgetter
from utils.q import calcQ
import numpy as np
from utils.numerical_processing import min_max


def plotQs(filePath, delay_s):
    with open(filePath, "rb") as f:
        docs = pickle.load(f)

    docs = sorted(  # datetimeでソート
        docs,
        key=lambda doc: datetime.datetime.fromisoformat(doc["_source"]["JPtime"])
        + datetime.timedelta(hours=9),
    )
    times = list(
        map(
            lambda doc: datetime.datetime.fromisoformat(doc["_source"]["JPtime"]),
            docs,
        )
    )
    qs = list(map(lambda doc: doc["_source"]["solarIrradiance(kw/m^2)"], docs))

    indexes = [0]
    prevDt = times[0]
    for i, time in enumerate(
        times
    ):  # 全てのdocumentをグラフ描画すると負荷がかかるので1分あたり1ドキュメントになるようデータを削る
        if prevDt.isoformat()[:16] != time.isoformat()[:16]:
            indexes.append(i)
        prevDt = time

    dtsByMinute = itemgetter(*indexes)(times)
    print(f"len(dtsByMinute): {len(dtsByMinute)}")
    qs = itemgetter(*indexes)(qs)
    plt.plot(dtsByMinute, min_max(qs), label="実測値")  # 実データをプロット

    qs_calc = list(  # 理論値を計算(kW/m^2に変換している)
        map(
            lambda dt: max(
                calcQ((dt + datetime.timedelta(seconds=delay_s)), 33.82794, 132.75093)
                / 1000,
                0,
            ),
            dtsByMinute,
        )
    )
    qs_calc_scaled = min_max(qs_calc)
    # qs_calc_scaled = list(map(lambda q: q * (max(qs) / max(qs_calc)), qs_calc))
    plt.plot(  # 理論値をプロット
        dtsByMinute,
        qs_calc_scaled,  # 縦軸のスケールを実測値と揃えている
        label="理論値",
    )

    # x 軸のラベルを設定する。
    plt.xlabel("日時")
    # y 軸のラベルを設定する。
    plt.ylabel("日射量[kW/m^2]")
    plt.legend()
    plt.show()
