import datetime
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import os
import argparse
import numpy as np
from utils.q import calc_qs_kw_v2
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates


def correlate_full(x, y):
    n = x.size
    m = y.size
    result = np.array([0] * (n + m - 1))
    for i in range(n):
        for j in range(m):
            result[i + j] += x[i] * y[j]
    return result


# > python3 partial_corr.py -dt 2022/09/30
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    args = parser.parse_args()

    year, month, date = args.dt.split("/")
    from_dt = datetime.datetime(
        int(year),
        int(month),
        int(date),
    )

    diff_days = 1.0
    dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days, True)
    dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

    calced_q_all = calc_qs_kw_v2(
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        altitude=25.720,
        surface_tilt=22,
        surface_azimuth=185,
    )

    axes = [plt.subplots()[1] for _ in range(1)]

    # 08:30 ~
    # mask = dt_all > datetime.datetime(int(year), int(month), int(date), 8, 30, 0)

    # 08:30 ~ 15:30
    # mask = (
    #     (datetime.datetime(int(year), int(month), int(date), 15, 30, 0)) > dt_all
    # ) & (dt_all > datetime.datetime(int(year), int(month), int(date), 8, 30, 0))

    # 1日全て
    mask = (
        (
            datetime.datetime(int(year), int(month), int(date), 0, 0, 0)
            + datetime.timedelta(days=1)
        )
        > dt_all
    ) & (dt_all > datetime.datetime(int(year), int(month), int(date), 0, 0, 0))

    masked_q_all = q_all[mask]
    masked_q_all_mean0 = masked_q_all - masked_q_all.mean()

    masked_calc_q_all = calced_q_all[mask]
    masked_calc_q_all_mean0 = masked_calc_q_all - masked_calc_q_all.mean()

    masked_dt_all = dt_all[mask]

    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(masked_dt_all)

    corr = np.correlate(masked_q_all_mean0, masked_calc_q_all_mean0, "full")
    estimated_delay = corr.argmax() - (len(masked_calc_q_all_mean0) - 1)

    print(f"ずれ時間: {estimated_delay}[s]")

    axes[0].plot(
        unified_dates,
        masked_q_all,
        label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[0].plot(
        unified_dates,
        masked_calc_q_all,
        label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
        linestyle="dashed",
        color=colorlist[1],
    )
    axes[0].set_title(f"ずれ時間: {estimated_delay}[s]")
    axes[0].set_xlabel("時刻")
    axes[0].set_ylabel("日射量[kW/m^2]")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[0].legend()

    plt.show()
