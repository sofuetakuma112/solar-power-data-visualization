import datetime
import json
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
from utils.corr import calc_delay
from utils.es.load import load_q_and_dt_for_period
import argparse
import numpy as np
from utils.q import Q
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from utils.init_matplotlib import init_rcParams, figsize_px_to_inch

# > python3 find_drop.py -dt 2022/04/08 -surface_tilt 28 -surface_azimuth 178.28
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付
    parser.add_argument(
        "-model", type=str, default="isotropic"
    )  # 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    parser.add_argument("-surface_tilt", type=int, default=22)
    parser.add_argument("-surface_azimuth", type=float, default=185.0)
    args = parser.parse_args()

    year, month, day = args.dt.split("/")
    from_dt = datetime.datetime(
        int(year),
        int(month),
        int(day),
    )

    diff_days = 1.0
    dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
    dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

    # 差分を計算
    diff = np.diff(q_all)

    # 最大差分のインデックスを取得
    max_diff_index = np.argmax(diff)

    print("最大差分のインデックス:", max_diff_index)

    # タイムスタンプとデータを結合
    time_series = pd.Series(q_all, index=dt_all)
    # 差分を計算
    diff = time_series.diff().dropna()
    # 最大差分のインデックス（タイムスタンプ）を取得
    max_diff_timestamp = diff.idxmax()
    print("最大差分のタイムスタンプ:", max_diff_timestamp)

    # 時間帯によって傾きが変わる時系列データをグループ化し、
    # 各グループの始点と終点を列挙するPythonコードは以下のようになります。

    # 傾きのしきい値
    threshold = 0.0015

    # 傾きを計算
    slope = np.diff(q_all)

    # グループ化のためのラベル
    group_label = 0
    group_labels = [group_label]

    # 傾きが似ている部分のグルーピング
    for i in range(1, len(slope)):
        if abs(slope[i] - slope[i - 1]) > threshold:
            group_label += 1
        group_labels.append(group_label)

    # 各グループの始点と終点を列挙
    group_labels = pd.Series(group_labels, index=dt_all[:-1])
    groups = group_labels.groupby(group_labels)
    for group, group_indices in groups:
        start = group_indices.index[0]
        end = group_indices.index[-1]
        print(f"Group {group}: Start = {start}, End = {end}")

    figsize_inch = figsize_px_to_inch(np.array([1920, 1080]))
    plt.rcParams = init_rcParams(plt.rcParams, 16, figsize_inch)

    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(dt_all)

    axes = [plt.subplots()[1] for i in range(2)]
    # axes[0].plot(diff, marker="o", linestyle="-", label="Diff")
    axes[0].plot(
        unified_dates,
        q_all,
        label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[0].set_xlabel("時刻")
    axes[0].set_ylabel("日射量 [kW/m$^2$]")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].legend()

    # ハイライトするデータ点をプロット
    matching_idx = np.where(dt_all == max_diff_timestamp)[0]
    highlight_value = q_all[matching_idx]
    axes[0].plot(
        unified_dates[matching_idx],
        highlight_value,
        "ro",
        markersize=4,
        label="Highlighted Data",
        color=colorlist[1],
    )

    # カラーマップを取得
    cmap = plt.get_cmap("tab20")

    # グループごとに色を変えてプロット
    for group, group_indices in groups:
        start = group_indices.index[0]
        end = group_indices.index[-1] + pd.Timedelta(seconds=1)
        axes[1].plot(time_series[start:end], label=f"Group {group}", color=cmap(group))

    axes[1].set_xlabel("時刻")
    axes[1].set_ylabel("日射量 [kW/m$^2$]")

    plt.legend()
    plt.show()
