import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.corr import calc_delay
from utils.es.load import load_q_and_dt_for_period
import os
import argparse
import numpy as np
from utils.q import Q, calc_q_kw
from utils.correlogram import (
    unify_deltas_between_dts_v2,
)
import matplotlib.dates as mdates
from utils.colors import colorlist

FONT_SIZE = 14

# > python3 plot_multiple_q.py -dts 2022/09/30 2022/04/08 2022/11/20 2022/05/03 2022/05/18 2022/10/30
# > python3 plot_multiple_q.py -dts 2022/04/08 -rv
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dts", type=str, nargs="*")  # グラフ描画したい日付のリスト
    parser.add_argument("-rv", "--real_value", action="store_true")  # 実測値を表示するか
    parser.add_argument("-tv", "--theoretical_value", action="store_true")  # 理論値を表示するか
    parser.add_argument("-sd", "--save_daily", action="store_true")

    args = parser.parse_args()
    dts = np.array(args.dts)

    sorted_indexes = np.vectorize(
        lambda dt: datetime.datetime.strptime(dt, "%Y/%m/%d")
    )(dts).argsort()
    dts = dts[sorted_indexes]

    figsize_px = np.array([1280, 720])
    dpi = 100
    figsize_inch = figsize_px / dpi

    axes = [plt.subplots(figsize=figsize_inch, dpi=dpi)[1] for _ in range(2)]

    def plot_daily(dt_str, i):
        if args.save_daily:
            _ax = [plt.subplots()[1] for _ in range(1)][0]

        year, month, date = dt_str.split("/")

        from_dt = datetime.datetime(
            int(year),
            int(month),
            int(date),
        )

        diff_days = 1.0
        dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
        dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

        unified_dates = np.vectorize(
            lambda dt: datetime.datetime(
                2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
            )
        )(dt_all)

        # 日毎に理論値と実測値の差を計算する
        q = Q()
        surface_tilt = 28
        surface_azimuth = 178.28

        calced_q_all = q.calc_qs_kw_v2(
            dt_all,
            latitude=33.82794,
            longitude=132.75093,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            model="isotropic",
        )

        (
            corr,
            estimated_delay,
        ) = calc_delay(calced_q_all, q_all)

        diffs = calced_q_all - q_all

        if args.real_value:
            axes[0].plot(
                unified_dates,
                q_all,
                label=dt_all[0].strftime("%Y-%m-%d"),
                color=colorlist[i],
            )
        if args.theoretical_value:
            axes[0].plot(
                unified_dates,
                calced_q_all,
                label=f"理論値: {dt_all[0].strftime('%Y-%m-%d')}",
                linestyle="dashed" if args.real_value else "solid",
                color=colorlist[i + 1],
            )

        axes[0].set_xlabel("日時", fontsize=FONT_SIZE)
        axes[0].set_ylabel("日射量 [kW/m$^2$]", fontsize=FONT_SIZE)
        axes[0].set_title(f"ずれ時間: {estimated_delay}s", fontsize=FONT_SIZE)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes[0].legend(fontsize=FONT_SIZE)
        axes[0].tick_params(axis='both', which='major', labelsize=FONT_SIZE)

        axes[1].plot(
            unified_dates,
            diffs,
            label=dt_all[0].strftime("%Y-%m-%d"),
            color=colorlist[i],
        )
        axes[1].set_xlabel("日時")
        axes[1].set_ylabel("理論値 - 実測値 [kW/m$^2$]")
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes[1].legend()

        if args.save_daily:
            # 特定の日のグラフを_axに描画
            _ax.plot(
                unified_dates,
                q_all,
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

    [plot_daily(dt_str, i) for i, dt_str in enumerate(dts)]  # np.vectorizeで書くとバグる

    plt.show()
    # plt.savefig("./graph.png")
