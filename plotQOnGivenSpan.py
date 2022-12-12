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


def dt_to_hours(dt):
    return dt.days * 24 + (dt.seconds + dt.microseconds / 1000000) / 60 / 60


# > python3 plotQOnGivenSpan.py -fd 2022/04/01 -ft 00:00:00 -td 2022/05/01 -tt 00:00:00
if __name__ == "__main__":
    # args = sys.argv
    # from_date = args[1].split("/")
    # from_time = args[2].split(":")
    # to_date = args[3].split("/")
    # to_time = args[4].split(":")

    parser = argparse.ArgumentParser(description="add two integer")
    parser.add_argument("-fd", type=str)  # from の日付
    parser.add_argument("-ft", type=str)  # fromの時間
    parser.add_argument("-td", type=str)  # toの日付
    parser.add_argument("-tt", type=str)  # toの時間
    parser.add_argument(
        "-tv", "--theoretical_value", action="store_true"
    )  # 理論値も合わせて表示するか

    args = parser.parse_args()

    from_date = args.fd.split("/")
    from_time = args.ft.split(":")
    to_date = args.td.split("/")
    to_time = args.tt.split(":")

    from_dt = datetime.datetime(
        int(from_date[0]),
        int(from_date[1]),
        int(from_date[2]),
        int(from_time[0]),
        int(from_time[1]),
        int(from_time[2]),
    )

    to_dt = datetime.datetime(
        int(to_date[0]),
        int(to_date[1]),
        int(to_date[2]),
        int(to_time[0]),
        int(to_time[1]),
        int(to_time[2]),
    )

    diff = to_dt - from_dt
    diff_days = dt_to_hours(diff) / 24

    dt_all, Q_all = load_q_and_dt_for_period(from_dt, diff_days, True)

    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)

    dt_all = np.array(dt_all)
    Q_all = np.array(Q_all)

    print(f"len(dt_all): {len(dt_all)}")
    print(f"len(Q_all): {len(Q_all)}")

    axes = [plt.subplots()[1] for i in range(1)]

    axes[0].plot(dt_all, Q_all, label="実測値")  # 実データをプロット
    axes[0].set_xlabel("日時")
    axes[0].set_ylabel("日射量[kW/m^2]")
    axes[0].set_title(f"{from_dt} ~ {to_dt}")
    axes[0].set_xlim(from_dt, to_dt)
    if args.theoretical_value:
        axes[0].plot(dt_all, np.vectorize(calc_q_kw)(dt_all), label="理論値")
    plt.show()
