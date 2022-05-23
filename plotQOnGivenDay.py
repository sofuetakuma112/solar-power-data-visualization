import pickle
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime
from operator import itemgetter
from com_global import calcQ


def plotQs(filePath):
    with open(filePath, "rb") as f:
        docs = pickle.load(f)

    docs = sorted( # datetimeでソート
        docs,
        key=lambda doc: datetime.datetime.fromisoformat(doc["_source"]["utctime"])
        + datetime.timedelta(hours=9),
    )
    times = list( # 時刻をUTCからJSTに変換する
        map(
            lambda doc: datetime.datetime.fromisoformat(doc["_source"]["utctime"])
            + datetime.timedelta(hours=9),
            docs,
        )
    )
    qs = list(map(lambda doc: doc["_source"]["solarIrradiance(kw/m^2)"], docs))

    indexes = [0]
    prevDt = times[0]
    for i, time in enumerate(times): # 全てのdocumentをグラフ描画すると負荷がかかるので1分あたり1ドキュメントになるようデータを削る
        if prevDt.isoformat()[:16] != time.isoformat()[:16]:
            indexes.append(i)
        prevDt = time

    dtsByMinute = itemgetter(*indexes)(times)
    qs = itemgetter(*indexes)(qs)
    plt.plot( # 実データをプロット
        dtsByMinute,
        qs,
        label="実測値"
    )

    qs_theoretical = list( # 理論値を計算(kW/m^2に変換している)
        map(lambda dt: max(calcQ(dt, 33.82794, 132.75093) / 1000, 0), dtsByMinute)
    )
    plt.plot( # 理論値をプロット
        dtsByMinute,
        list(map(lambda q: q * (max(qs) / max(qs_theoretical)), qs_theoretical)), # 縦軸のスケールを実測値と揃えている
        label="理論値"
    )

    # x 軸のラベルを設定する。
    plt.xlabel("日時")

    # y 軸のラベルを設定する。
    plt.ylabel("日射量[kW/m^2]")
    plt.legend()
    plt.show()
