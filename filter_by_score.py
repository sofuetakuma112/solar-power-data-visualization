import argparse
import pandas as pd

import csv
import datetime
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import numpy as np
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from sklearn import preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str)  # グラフ描画したい日付のリスト
    args = parser.parse_args()

    DIR_PATH = "data/csv/scoring_measured_value"
    df = pd.read_csv(f"{DIR_PATH}/{args.csv}.csv")

    df["score"] = (1 / df["jaggedness"]) * df["area"]
    df = df.fillna(0)

    df = df.sort_values("score", ascending=False)

    dts = df.head(20)["dt"].to_numpy()

    print(f"dts: {dts}")

    # axes = [plt.subplots()[1] for _ in range(1)]
    # axes[0].plot(
    #     dt_all[first_index : last_index + 1],
    #     q_all[first_index : last_index + 1],
    #     label=f"実測値(カット済み): {dt_all[0].strftime('%Y-%m-%d')}",
    #     linestyle="dashed",
    #     color=colorlist[0],
    # )
    # axes[0].plot(
    #     dt_all,
    #     q_all,
    #     label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
    #     color=colorlist[1],
    # )
    # axes[0].set_xlabel("時刻")
    # axes[0].set_ylabel("日射量[kW/m^2]")
    # axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    # axes[0].legend()

    # plt.show()
