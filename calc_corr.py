import argparse
import datetime
from matplotlib import dates
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.corr import calc_delay
from utils.es.load import load_q_and_dt_for_period
import numpy as np
from utils.init_matplotlib import figsize_px_to_inch, init_rcParams
from utils.numerical_processing import min_max
from utils.q import Q, calc_q_kw
from utils.correlogram import (
    unify_deltas_between_dts_v2,
)
from utils.colors import colorlist


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


FONT_SIZE = 14

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    args = parser.parse_args()

    year, month, day = args.dt.split("/")
    from_dt = datetime.datetime(
        int(year),
        int(month),
        int(day),
    )
    to_dt = from_dt + datetime.timedelta(days=1)

    diff_days = 1.0
    dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
    dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

    lat = 33.82794
    lng = 132.75093

    q = Q()  # インスタンス作成時にDBへのコネクションを初期化
    calced_q_all = q.calc_qs_kw_v2(
        dt_all,
        latitude=lat,
        longitude=lng,
        surface_tilt=22,
        surface_azimuth=179.0,
        model="isotropic",
    )

    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(dt_all)

    (
        corr_with_real_and_calc,
        estimated_delay_with_real_and_calc,
    ) = calc_delay(calced_q_all, q_all)
    print(f"ずれ時間（実測値と計算値）: {estimated_delay_with_real_and_calc}[s]")

    figsize_inch = figsize_px_to_inch(np.array([1280, 720]))
    plt.rcParams = init_rcParams(plt.rcParams, FONT_SIZE, figsize_inch)

    axes = [plt.subplots()[1] for _ in range(2)]

    # 実測値と計算値
    axes[0].plot(
        unified_dates,
        q_all,
        label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[0].plot(
        unified_dates,
        calced_q_all,
        label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
        linestyle="dashed",
        color=colorlist[1],
    )
    axes[0].set_xlabel("時刻")
    axes[0].set_ylabel("日射量 [kW/m$^2$]")
    axes[0].xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    axes[0].legend()

    # 自作の計算式で計算データを求める
    calced_q_by_original = np.vectorize(lambda dt: calc_q_kw(dt, lat, lng))(dt_all)

    q_all = min_max(q_all)
    calced_q_by_original = min_max(calced_q_by_original)

    (
        corr_with_real_and_calc_by_original,
        estimated_delay_with_real_and_calc_by_original,
    ) = calc_delay(calced_q_by_original, q_all)
    print(f"ずれ時間（実測値と計算値（自作関数））: {estimated_delay_with_real_and_calc_by_original}[s]")

    axes[1].plot(
        unified_dates,
        q_all,
        label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[1].plot(
        unified_dates,
        calced_q_by_original,
        label=f"計算値（自作関数）: {dt_all[0].strftime('%Y-%m-%d')}",
        linestyle="dashed",
        color=colorlist[1],
    )
    axes[1].set_xlabel("時刻")
    axes[1].set_ylabel("日射量 [kW/m$^2$]")
    axes[1].xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    axes[1].legend()

    plt.show()
