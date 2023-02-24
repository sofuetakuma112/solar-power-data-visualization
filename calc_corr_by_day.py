import argparse
import pandas as pd

import csv
import datetime
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.corr import calc_delay
from utils.date import mask_from_into_dt, mask_to_into_dt
from utils.es.load import load_q_and_dt_for_period
import numpy as np
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from sklearn import preprocessing

from utils.q import Q

mask_from_tos = {
    "2022/06/02": {
        "mask_from": "06:35",
        "mask_to": "17:35",
    },
    "2022/06/03": {
        "mask_from": "06:35",
        "mask_to": "17:35",
    },
    "2022/04/08": {
        "mask_from": "07:20",
        "mask_to": "17:10",
    },
    "2022/05/18": {
        "mask_from": "06:40",
        "mask_to": "17:30",
    },
    "2022/05/22": {
        "mask_from": "06:40",
        "mask_to": "17:30",
    },
    "2022/05/03": {
        "mask_from": "06:50",
        "mask_to": "17:30",
    },
    "2022/10/20": {
        "mask_from": "08:20",
        "mask_to": "15:20",
    },
    "2022/09/30": {
        "mask_from": "08:00",
        "mask_to": "15:50",
    },
    "2022/11/09": {
        "mask_from": "08:35",
        "mask_to": "15:10",
    },
    "2022/08/29": {
        "mask_from": "00:00",
        "mask_to": "23:59",
    },
}


def cleanup(df):
    OUTPUT_CSV_DIR_PATH = "data/csv/calc_corr_by_day"
    if not os.path.exists(OUTPUT_CSV_DIR_PATH):
        os.makedirs(OUTPUT_CSV_DIR_PATH)
    df.to_csv(f"{OUTPUT_CSV_DIR_PATH}/result.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-head", type=int)  # DataFrameの上位何件のみを使用するか
    parser.add_argument("-masking_strategy", type=str, default="replace_zero")
    args = parser.parse_args()

    DIR_PATH = "data/csv/filter_by_score"
    df = pd.read_csv(f"{DIR_PATH}/score.csv")

    if args.head == None:
        dts = df["dt"].to_numpy()
    else:
        dts = df.head(args.head)["dt"].to_numpy()

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
    df = pd.DataFrame(
        [],
        columns=columns,
    )

    q = Q()
    surface_tilt = 28
    surface_azimuth = 178.28
    for dt in dts:
        try:
            year, month, date = dt.split("/")
            from_dt = datetime.datetime(
                int(year),
                int(month),
                int(date),
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
                print("マスクの開始時刻を入力してください")
                mask_from = input()
                mask_from = mask_from_into_dt(mask_from, year, month, date)

                print("マスクの終了時刻を入力してください")
                mask_to = input()
                mask_to = mask_to_into_dt(mask_to)

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
                    "mask_from": mask_from.strftime('%Y/%m/%d %H:%M:%S'),
                    "mask_to": mask_to.strftime('%Y/%m/%d %H:%M:%S'),
                    "surface_tilt": surface_tilt,
                    "surface_azimuth": surface_azimuth,
                }
            )
            df = df.append(new_row, ignore_index=True)
        except:
            cleanup(df)

    cleanup(df)
