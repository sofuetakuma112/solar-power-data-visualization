import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import os
import json
from utils.q import calc_q_kw
import numpy as np
from utils.correlogram import (
    unify_deltas_between_dts,
)


# FIXME: 2022-09-22でバグってる
if __name__ == "__main__":
    json_open = open("data/json/sorted_diffs.json", "r")
    diff_and_dates = json.load(json_open)

    for diff_and_date in diff_and_dates:
        diff, date = diff_and_date
        days_len = 1.0
        fromDt = datetime.datetime.strptime(date, "%Y-%m-%d")
        toDt = fromDt + datetime.timedelta(days=days_len)

        dt_all, Q_all = load_q_and_dt_for_period(fromDt, days_len, True)
        dt_all, Q_all = unify_deltas_between_dts(dt_all, Q_all)
        dt_all = np.array(dt_all)
        Q_all = np.array(Q_all)

        q_calc_all = np.vectorize(calc_q_kw)(dt_all)

        corr = np.correlate(Q_all, q_calc_all, "full")
        estimated_delay = corr.argmax() - (len(q_calc_all) - 1)

        axes = [plt.subplots()[1] for i in range(1)]

        axes[0].plot(dt_all, Q_all, label="実測値")  # 実データをプロット
        axes[0].plot(
            dt_all,
            q_calc_all,
            label="理論値",
            linestyle="dashed",
        )
        axes[0].set_xlabel("日時")
        axes[0].set_ylabel("日射量[kW/m^2]")
        axes[0].set_title(
            f"{fromDt.strftime('%Y-%m-%d')} estimated_delay: {estimated_delay}"
        )
        axes[0].set_xlim(fromDt, toDt)
        axes[0].legend()

        dir = "./images/daily_q"
        if not os.path.exists(dir):
            os.makedirs(dir)

        plt.savefig(f"{dir}/{diff}.png")
