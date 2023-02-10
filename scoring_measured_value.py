import csv
import datetime
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import argparse
import numpy as np
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from sklearn import preprocessing


def calculate_waveform_area(waveform):
    # 波形の面積を計算
    area = np.sum(waveform) * (1.0 / waveform.size)
    return area


# ギザギザさ
def smoothness_score(waveform):
    # 差分計算
    diff = np.diff(waveform)
    # 差分の絶対値の平均
    score = np.sum(np.abs(diff))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    args = parser.parse_args()

    year, month, date = args.dt.split("/")
    start_dt = datetime.datetime(
        int(year),
        int(month),
        int(date),
    )

    from_dt = start_dt

    now = datetime.datetime.now()

    DIR_PATH = "data/csv/scoring_measured_value"
    if not os.path.exists(DIR_PATH):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(DIR_PATH)

    header = ["dt", "from_time", "to_time", "jaggedness", "area"]
    with open(f"{DIR_PATH}/data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # TODO: ElasticSearchの全期間を対象に1日ずつ計算する
        while from_dt.date() < now.date():
            print(f"from_dt: {from_dt}")

            diff_days = 1.0
            dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
            dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

            # 日射量が0になっている箇所を切り落とす
            # indexes = np.argwhere(q_all > 0.003)
            # first_index = indexes[0][0]
            # last_index = indexes[-1][0]
            # cut_q_all = q_all[first_index : last_index + 1]

            jaggedness = smoothness_score(preprocessing.minmax_scale(q_all))
            area = calculate_waveform_area(q_all)

            print(f"ギザギザさ: {jaggedness}")
            print(f"面積: {area}")

            writer.writerow(
                [from_dt.strftime("%Y/%m/%d"), "00:00:00", "23:59:59", jaggedness, area]
            )

            from_dt = from_dt + datetime.timedelta(days=1)
