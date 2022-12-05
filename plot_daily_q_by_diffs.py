import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import loadQAndDtForAGivenPeriod
import os
import json
from utils.q import calc_q_kw
import numpy as np


if __name__ == "__main__":
    json_open = open("data/json/sorted_diffs.json", "r")
    diff_and_dates = json.load(json_open)

    for diff_and_date in diff_and_dates:
        fromDt = datetime.datetime.strptime(diff_and_date[-1], "%Y-%m-%d")
        toDt = fromDt + datetime.timedelta(days=1)

        # FIXME: 欠損データを0埋めする処理を実装する
        dt_all, Q_all = loadQAndDtForAGivenPeriod(fromDt, toDt, True)

        dt_all = np.array(dt_all)
        Q_all = np.array(Q_all)

        corr = np.correlate(Q_all, np.vectorize(calc_q_kw)(dt_all), "full")
        estimated_delay = corr.argmax() - (len(Q_all) - 1)

        axes = [plt.subplots()[1] for i in range(1)]

        axes[0].plot(dt_all, Q_all, label="実測値")  # 実データをプロット
        axes[0].set_xlabel("日時")
        axes[0].set_ylabel("日射量[kW/m^2]")
        axes[0].set_title(f"{fromDt.strftime('%Y-%m-%d')} estimated_delay: {estimated_delay}")
        axes[0].set_xlim(fromDt, toDt)
        # plt.show()

        dir = "./images/daily_q"
        if not os.path.exists(dir):
            os.makedirs(dir)

        plt.savefig(
            f"{dir}/{fromDt.strftime('%Y-%m-%d')}.png"
        )
