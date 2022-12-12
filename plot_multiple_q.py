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


# > python3 plot_multiple_q.py -dts 2022/09/30 2022/04/08 2022/11/20 2022/05/03 2022/05/18 2022/10/30
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dts", type=str, nargs="*")  # グラフ描画したい日付のリスト
    parser.add_argument(
        "-tv", "--theoretical_value", action="store_true"
    )  # 理論値も合わせて表示するか

    args = parser.parse_args()
    dts = np.array(args.dts)

    axes = [plt.subplots()[1] for _ in range(1)]

    def plot_daily(dt_str):
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

        axes[0].plot(
            unified_dates,
            Q_all,
            label=dt_all[0].strftime("%Y-%m-%d"),
        )  # 実データをプロット
        if args.theoretical_value:
            axes[0].plot(unified_dates, np.vectorize(calc_q_kw)(dt_all), label="理論値")

    axes[0].set_xlabel("日時")
    axes[0].set_ylabel("日射量[kW/m^2]")

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    [plot_daily(dt_str) for dt_str in dts] # np.vectorizeで書くとバグる

    axes[0].legend()

    plt.show()
