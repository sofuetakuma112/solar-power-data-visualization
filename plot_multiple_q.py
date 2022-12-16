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
    unifyDeltasBetweenDts,
)
import matplotlib.dates as mdates


def dt_to_hours(dt):
    return dt.days * 24 + (dt.seconds + dt.microseconds / 1000000) / 60 / 60


def time_to_seconds(t):
    return (t.hour * 60 + t.minute) * 60 + t.second


colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

# > python3 plot_multiple_q.py -dts 2022/09/30 2022/04/08 2022/11/20 2022/05/03 2022/05/18 2022/10/30
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dts", type=str, nargs="*")  # グラフ描画したい日付のリスト
    parser.add_argument(
        "-tv", "--theoretical_value", action="store_true"
    )  # 理論値も合わせて表示するか
    parser.add_argument(
        "-sd", "--save_daily", action="store_true"
    )  # 理論値も合わせて表示するか

    args = parser.parse_args()
    dts = np.array(args.dts)

    ax = [plt.subplots()[1] for _ in range(1)][0]

    def plot_daily(dt_str, i):
        if args.save_daily:
            _ax = [plt.subplots()[1] for _ in range(1)][0]

        dt = dt_str.split("/")

        from_dt = datetime.datetime(
            int(dt[0]),
            int(dt[1]),
            int(dt[2]),
        )

        diff_days = 1.0
        dt_all, Q_all = load_q_and_dt_for_period(from_dt, diff_days, True)
        dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)

        dt_all = np.array(dt_all)
        Q_all = np.array(Q_all)

        unified_dates = np.vectorize(
            lambda dt: datetime.datetime(
                2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
            )
        )(dt_all)

        # # 複数の日のグラフを共通のaxに描画
        ax.plot(
            unified_dates,
            Q_all,
            label=dt_all[0].strftime("%Y-%m-%d"),
            color=colorlist[i],
        )
        if args.theoretical_value:
            ax.plot(
                unified_dates,
                np.vectorize(calc_q_kw)(dt_all),
                label=f"理論値: {dt_all[0].strftime('%Y-%m-%d')}",
                linestyle="dashed",
                color=colorlist[i],
            )

        if args.save_daily:
            # 特定の日のグラフを_axに描画
            _ax.plot(
                unified_dates,
                Q_all,
                label=dt_all[0].strftime("%Y-%m-%d"),
                color=colorlist[i],
            )
            if args.theoretical_value:
                _ax.plot(
                    unified_dates,
                    np.vectorize(calc_q_kw)(dt_all),
                    label=f"理論値: {dt_all[0].strftime('%Y-%m-%d')}",
                    linestyle="dashed",
                    color=colorlist[i],
                )
            _ax.set_xlabel("日時")
            _ax.set_ylabel("日射量[kW/m^2]")
            _ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            _ax.legend()

            dir = "./images/daily_q_with_tv"
            if not os.path.exists(dir):
                os.makedirs(dir)

            date_str = from_dt.strftime("%Y-%m-%d")
            plt.savefig(f"{dir}/{date_str}.png")

    ax.set_xlabel("日時")
    ax.set_ylabel("日射量[kW/m^2]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    [plot_daily(dt_str, i) for i, dt_str in enumerate(dts)]  # np.vectorizeで書くとバグる

    ax.legend()

    plt.show()
    # plt.savefig("./graph.png")
