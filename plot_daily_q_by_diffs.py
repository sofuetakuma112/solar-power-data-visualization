import datetime
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import os
import json
from utils.q import calc_qs_kw_v2
import numpy as np
from utils.correlogram import (
    unify_deltas_between_dts_v2,
)
from tqdm import tqdm

if __name__ == "__main__":
    json_open = open("data/json/sorted_diffs.json", "r")
    diff_and_dates = json.load(json_open)

    for i, diff_and_date in enumerate(tqdm(diff_and_dates)):
        diff, date = diff_and_date
        days_len = 1.0
        from_dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        to_dt = from_dt + datetime.timedelta(days=days_len)

        print(f"current dt: {date}")
        dir = "./images/daily_q"
        if not os.path.exists(dir):
            os.makedirs(dir)
        fig_file_path = f"{dir}/{str(i).zfill(4)}: {from_dt.strftime('%Y-%m-%d')}.png"
        if os.path.isfile(fig_file_path):
            continue

        dt_all, q_all = load_q_and_dt_for_period(from_dt, days_len, True)
        dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

        calced_q_all = calc_qs_kw_v2(
            dt_all,
            latitude=33.82794,
            longitude=132.75093,
            altitude=25.720,
            surface_tilt=22,
            surface_azimuth=185,
            model="isotropic",
        )

        q_all_mean0 = q_all - q_all.mean()
        calc_q_all_mean0 = calced_q_all - calced_q_all.mean()

        corr = np.correlate(q_all_mean0, calc_q_all_mean0, "full")
        estimated_delay = corr.argmax() - (len(calc_q_all_mean0) - 1)

        axes = [plt.subplots()[1] for _ in range(1)]

        axes[0].plot(dt_all, q_all, label="実測値")  # 実データをプロット
        axes[0].plot(
            dt_all,
            calced_q_all,
            label="理論値",
            linestyle="dashed",
        )
        axes[0].set_xlabel("日時")
        axes[0].set_ylabel("日射量[kW/m^2]")
        axes[0].set_title(f"{from_dt.strftime('%Y-%m-%d')} ずれ時間: {estimated_delay}")
        axes[0].set_xlim(from_dt, to_dt)
        axes[0].legend()

        plt.savefig(fig_file_path)
