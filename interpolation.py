import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import os
import argparse
import numpy as np
from utils.q import calc_q_kw
from utils.correlogram import (
    unify_deltas_between_dts,
)
from utils.colors import colorlist
import matplotlib.dates as mdates

# > python3 plot_multiple_q.py -dt 2022/09/30
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    args = parser.parse_args()

    dt_str = args.dt

    year, month, date = dt_str.split("/")

    from_dt = datetime.datetime(
        int(year),
        int(month),
        int(date),
    )

    diff_days = 1.0
    dt_all, Q_all = load_q_and_dt_for_period(from_dt, diff_days, True)
    dt_all, Q_all = unify_deltas_between_dts(dt_all, Q_all)

    dt_all = np.array(dt_all)
    Q_all = np.array(Q_all)

    calced_q_all = np.vectorize(calc_q_kw)(dt_all)

    # グラフ表示用の日付データ列
    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(dt_all)

    # 理論値を段階的に実測値に近づけていく
    num_interp = 5  # インターポレーションする数
    interp_steps = np.linspace(0, 1, num_interp)

    axes = [plt.subplots()[1] for _ in range(num_interp)]
    for i, alpha in enumerate(interp_steps):
        interp_c_all = calced_q_all + (Q_all - calced_q_all) * alpha

        corr = np.correlate(Q_all, interp_c_all, "full")
        estimated_delay = corr.argmax() - (len(interp_c_all) - 1)

        print(estimated_delay)

        axes[i].plot(
            unified_dates,
            Q_all,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[i],
        )
        axes[i].plot(
            unified_dates,
            interp_c_all,
            label=f"理論値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[i],
        )
        axes[i].set_xlabel("時刻")
        axes[i].set_ylabel("日射量[kW/m^2]")
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes[i].legend()

    plt.show()
