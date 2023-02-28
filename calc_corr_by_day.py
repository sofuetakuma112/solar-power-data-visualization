import argparse
import json
import re
import pandas as pd

import csv
import datetime
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.corr import calc_delay
from utils.date import mask_from_into_dt, mask_to_into_dt, str2datetime
from utils.es.load import load_q_and_dt_for_period
import numpy as np
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from sklearn import preprocessing

from utils.q import Q


def input_with_validate(message):
    while True:
        print(message, end="")
        input_text = input()
        pattern = "^\d{2}:\d{2}$"
        if re.match(pattern, input_text):
            return input_text
        else:
            print("XX:XXの形式で入力してください")
            continue


OUTPUT_CSV_DIR_PATH = "data/csv/calc_corr_by_day"
OUTPUT_CSV_FILE_PATH = f"{OUTPUT_CSV_DIR_PATH}/result.csv"

USER_INPUT_JSON_FILE_PATH = f"data/json/calc_corr_by_day/user_input"

def cleanup(df, mask_from_tos):
    if not os.path.exists(OUTPUT_CSV_DIR_PATH):
        os.makedirs(OUTPUT_CSV_DIR_PATH)

    df.to_csv(OUTPUT_CSV_FILE_PATH)
    with open(USER_INPUT_JSON_FILE_PATH, 'w') as f:
        json.dump(mask_from_tos, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-head", type=int)  # DataFrameの上位何件のみを使用するか
    parser.add_argument("-masking_strategy", type=str, default="replace_zero")
    args = parser.parse_args()

    json_open = open(USER_INPUT_JSON_FILE_PATH, "r")
    mask_from_tos = json.load(json_open)

    DIR_PATH = "data/csv/filter_by_score"
    score_df = pd.read_csv(f"{DIR_PATH}/score.csv")

    if args.head == None:
        dts = score_df["dt"].to_numpy()
    else:
        dts = score_df.head(args.head)["dt"].to_numpy()

    # 日付ごとに相互相関を計算してズレ時間を求める
    columns = [
        "dt",
        "estimated_delay",
        "partial_estimated_delay",
        "mask_from",
        "mask_to",
        "surface_tilt",
        "surface_azimuth",
    ]

    # data/csv/calc_corr_by_day/result.csv が既に存在する場合はそれを読み込む
    if os.path.isfile(OUTPUT_CSV_FILE_PATH):
        df = pd.read_csv(OUTPUT_CSV_FILE_PATH, index_col=0)
    else:
        df = pd.DataFrame(
            [],
            columns=columns,
        )
    n_rows = len(df.index)

    q = Q()
    surface_tilt = 28
    surface_azimuth = 178.28
    for i, dt in enumerate(dts):
        try:
            year, month, date = dt.split("/")
            from_dt = datetime.datetime(
                int(year),
                int(month),
                int(date),
            )

            fig_dir_path = f"{OUTPUT_CSV_DIR_PATH}/figures"  # 図の保存先ディレクトリ
            if not os.path.exists(fig_dir_path):
                os.makedirs(fig_dir_path)
            fig_file_path = (  # 図の保存パス
                f"{fig_dir_path}/{str(i).zfill(4)}: {from_dt.strftime('%Y-%m-%d')}.png"
            )

            print(from_dt)
            diff_days = 1.0
            dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
            dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)
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
            # 1. プロットする
            has_mask_pair = dt in mask_from_tos

            if not has_mask_pair:
                axes = [plt.subplots()[1] for _ in range(1)]
                axes[0].plot(
                    dt_all,
                    q_all,
                    label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
                    color=colorlist[0],
                )
                axes[0].set_title(f"ずれ時間={estimated_delay}[s]")
                axes[0].set_xlabel("時刻")
                axes[0].set_ylabel("日射量[kW/m^2]")
                axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                axes[0].legend()
                plt.show()
            # 2. mask_fromとmask_toを標準入力で受け取る
            if has_mask_pair:
                # 既にmask_from, mask_toが存在する
                mask_from = mask_from_into_dt(
                    mask_from_tos[dt]["mask_from"], year, month, date
                )
                mask_to = mask_from_into_dt(
                    mask_from_tos[dt]["mask_to"], year, month, date
                )
            else:
                # ユーザー入力で受け取る
                mask_from = input_with_validate("マスクの開始時刻を入力してください: ")
                mask_from = mask_from_into_dt(mask_from, year, month, date)

                mask_to = input_with_validate("マスクの終了時刻を入力してください: ")
                mask_to = mask_to_into_dt(mask_to, year, month, date)

            if i < n_rows:
                mask_from_csv = df.iloc[i]["mask_from"]
                mask_to_csv = df.iloc[i]["mask_to"]
            else:
                # 絶対に一致しない値で初期化しておく
                mask_from_csv = "1970/01/01 00:00:00"
                mask_to_csv = "1970/01/01 00:00:00"

            # print(f"mask_from: {mask_from}")
            # print(f"mask_to: {mask_to}")
            # print(f"mask_from_csv: {mask_from_csv}")
            # print(f"mask_to_csv: {mask_to_csv}")
            # print(f"fig_file_path: {fig_file_path}")

            if (
                mask_from == str2datetime(mask_from_csv, "/")
                and mask_to == str2datetime(mask_to_csv, "/")
                and os.path.isfile(fig_file_path)
            ):
                # CSVのmask_from, mask_toと一致して、画像も保存済みの場合
                continue

            # mask_from_tosに書き込む
            mask_from_tos[dt] = {
                "mask_from": mask_from.strftime("%H:%M"),
                "mask_to": mask_to.strftime("%H:%M"),
            }

            mask = (mask_from <= dt_all) & (dt_all < mask_to)

            # 3. マスク処理
            if args.masking_strategy == "drop":
                masked_q_all = q_all[mask]
                masked_calc_q_all = calced_q_all[mask]

                masked_dt_all = dt_all[mask]
            elif args.masking_strategy == "replace_zero":
                inverted_mask = np.logical_not(mask)
                np.putmask(q_all, inverted_mask, q_all * 0)
                np.putmask(calced_q_all, inverted_mask, calced_q_all * 0)

                masked_q_all = q_all
                masked_calc_q_all = calced_q_all

                masked_dt_all = dt_all
            else:
                raise ValueError("masking_strategyの値が不正")

            # 4. 相互相関を計算する
            (
                partial_corr,
                partial_estimated_delay,
            ) = calc_delay(masked_calc_q_all, masked_q_all)
            print(f"ずれ時間（実測値と計算値）: {partial_estimated_delay}[s]")
            # 5. dfに追加する
            new_row = pd.Series(
                {
                    "dt": dt,
                    "estimated_delay": estimated_delay,
                    "partial_estimated_delay": partial_estimated_delay,
                    "mask_from": mask_from.strftime("%Y/%m/%d %H:%M:%S"),
                    "mask_to": mask_to.strftime("%Y/%m/%d %H:%M:%S"),
                    "surface_tilt": surface_tilt,
                    "surface_azimuth": surface_azimuth,
                }
            )
            df = df.append(new_row, ignore_index=True)

            # 6. 指定したmask_fromからmask_toの範囲でプロットした実測値の図を画像として保存する
            if not os.path.isfile(fig_file_path):
                figsize_px = np.array([1280, 720])
                dpi = 100
                figsize_inch = figsize_px / dpi
                axes = [
                    plt.subplots(figsize=figsize_inch, dpi=dpi)[1] for _ in range(1)
                ]

                axes[0].plot(
                    masked_dt_all,
                    masked_q_all,
                    label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
                    color=colorlist[0],
                )
                axes[0].plot(
                    masked_dt_all,
                    masked_calc_q_all,
                    label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
                    linestyle="dashed",
                    color=colorlist[1],
                )

                axes[0].set_title(
                    f"{mask_from}〜{mask_to}, ずれ時間={partial_estimated_delay}[s]"
                )
                axes[0].set_xlabel("時刻")
                axes[0].set_ylabel("日射量[kW/m^2]")
                axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                axes[0].legend()

                plt.savefig(fig_file_path)

            plt.clf()
            plt.close()
        except Exception as e:
            print(e)
            cleanup(df, mask_from_tos)
            exit(1)

    cleanup(df, mask_from_tos)
