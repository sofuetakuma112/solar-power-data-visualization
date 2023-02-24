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

from utils.q import Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str)  # 計測データのCSVファイルの名前
    parser.add_argument("-head", type=int)  # DataFrameの上位何件のみを使用するか
    parser.add_argument("-only_csv", action="store_true") # スコアリングした結果をCSVに出力するだけにするか
    args = parser.parse_args()

    DIR_PATH = "data/csv/scoring_measured_value"
    df = pd.read_csv(f"{DIR_PATH}/{args.csv}.csv")

    df["score"] = (1 / df["jaggedness"]) * df["area"]
    df = df.fillna(0)

    df = df.sort_values("score", ascending=False)

    if args.head == None:
        dts = df["dt"].to_numpy()
    else:
        dts = df.head(args.head)["dt"].to_numpy()

    print(f"dts: {dts}")

    # csvで出力する
    OUTPUT_CSV_DIR_PATH = "data/csv/filter_by_score"
    if not os.path.exists(OUTPUT_CSV_DIR_PATH):
        os.makedirs(OUTPUT_CSV_DIR_PATH)
    df.to_csv(f'{OUTPUT_CSV_DIR_PATH}/score.csv')

    if args.only_csv:
        exit()

    dir = "./images/filter_by_score"
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i, dt in enumerate(dts):
        print(f"dt: {dt}")
        year, month, date = dt.split("/")
        from_dt = datetime.datetime(
            int(year),
            int(month),
            int(date),
        )

        fig_file_path = f"{dir}/{str(i).zfill(4)}: {from_dt.strftime('%Y-%m-%d')}.png"
        if os.path.isfile(fig_file_path):
            continue

        diff_days = 1.0
        dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
        dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

        q = Q()  # インスタンス作成時にDBへのコネクションを初期化
        calced_q_all = q.calc_qs_kw_v2(
            dt_all,
            latitude=33.82794,
            longitude=132.75093,
            surface_tilt=22,
            surface_azimuth=185,
            model="isotropic",
        )

        q_all_mean0 = q_all - q_all.mean()
        calc_q_all_mean0 = calced_q_all - calced_q_all.mean()

        corr = np.correlate(q_all_mean0, calc_q_all_mean0, "full")
        estimated_delay = corr.argmax() - (len(calc_q_all_mean0) - 1)

        figsize_px = np.array([1280, 720])
        dpi = 100
        figsize_inch = figsize_px / dpi
        axes = [plt.subplots(figsize=figsize_inch, dpi=dpi)[1] for _ in range(1)]

        axes[0].plot(
            dt_all,
            q_all,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[0].plot(
            dt_all,
            calced_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )

        df_row = df.query(f'dt == "{dt}"')
        jaggedness = round(float(df_row["jaggedness"]), 4)
        area = round(float(df_row["area"]), 4)
        score = round(float(df_row["score"]), 4)

        axes[0].set_title(
            f"ずれ時間={estimated_delay}[s] ぎざぎざ度={jaggedness} 面積={area} スコア={score}"
        )
        axes[0].set_xlabel("時刻")
        axes[0].set_ylabel("日射量[kW/m^2]")
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes[0].legend()

        plt.savefig(fig_file_path)
