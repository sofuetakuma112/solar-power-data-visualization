import datetime
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import os
import argparse
import numpy as np
from utils.q import Q, calc_q_kw
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates


def calc_delay(a, b):
    corr = np.correlate(a, b, "full")
    return [corr, corr.argmax() - (len(b) - 1)]


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
    parser.add_argument("-slide_seconds", type=int)
    parser.add_argument("-mask_from", type=str, default="00:00")
    parser.add_argument("-mask_to", type=str, default="24:00")
    args = parser.parse_args()

    year, month, date = args.dt.split("/")
    from_dt = datetime.datetime(
        int(year),
        int(month),
        int(date),
    )

    diff_days = 1.0
    dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
    dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

    # 昇順で並んでいるかテストする
    sort_indexes = np.argsort(dt_all)
    if not np.allclose(sort_indexes, np.arange(0, dt_all.size, 1)):
        raise ValueError("dt_allが時系列順で並んでいない")

    q = Q()  # インスタンス作成時にDBへのコネクションを初期化
    calced_q_all = q.calc_qs_kw_v2(
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=22,
        surface_azimuth=185,
        model="isotropic",
    )

    print(f"真のズレ時間: {args.slide_seconds}[s]")
    calced_q_all_slided = q.calc_qs_kw_v2(
        # dt_all + datetime.timedelta(seconds=870),
        dt_all + datetime.timedelta(seconds=args.slide_seconds),
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=22,
        surface_azimuth=185,
        model="isotropic",
    )
    # calced_q_all = np.vectorize(calc_q_kw)(dt_all)

    axes = [plt.subplots()[1] for _ in range(6)]

    mask_from_hour, mask_from_minute = args.mask_from.split(":")
    mask_from = datetime.datetime(
        int(year),
        int(month),
        int(date),
        int(mask_from_hour),
        int(mask_from_minute),
        int(0),
    )

    mask_to_hour, mask_to_minute = args.mask_to.split(":")
    if int(mask_to_hour) == 24:
        mask_to = datetime.datetime(
            int(year),
            int(month),
            int(date),
            int(0),
            int(mask_to_minute),
            int(0),
        ) + datetime.timedelta(days=1)
    else:
        mask_to = datetime.datetime(
            int(year),
            int(month),
            int(date),
            int(mask_to_hour),
            int(mask_to_minute),
            int(0),
        )

    print(f"{mask_from} ~ {mask_to}")

    # 08:30 ~
    # mask = dt_all > datetime.datetime(int(year), int(month), int(date), 8, 30, 0)

    # 08:30 ~ 15:30
    # mask = (
    #     (datetime.datetime(int(year), int(month), int(date), 15, 30, 0)) > dt_all
    # ) & (dt_all > datetime.datetime(int(year), int(month), int(date), 8, 30, 0))

    # 07:20 ~ 17:10
    # mask = (
    #     (datetime.datetime(int(year), int(month), int(date), 17, 10, 0)) > dt_all
    # ) & (dt_all > datetime.datetime(int(year), int(month), int(date), 7, 20, 0))

    # 1日全て
    mask = (mask_from <= dt_all) & (dt_all < mask_to)

    # マスク処理
    masked_q_all = q_all[mask]
    masked_calc_q_all = calced_q_all[mask]
    masked_calc_q_all_slided = calced_q_all_slided[mask]

    # 標準化
    # masked_q_all = (masked_q_all - np.mean(masked_q_all)) / np.std(masked_q_all)
    # masked_calc_q_all = (masked_calc_q_all - np.mean(masked_calc_q_all)) / np.std(
    #     masked_calc_q_all
    # )
    # masked_calc_q_all_slided = (
    #     masked_calc_q_all_slided - np.mean(masked_calc_q_all_slided)
    # ) / np.std(masked_calc_q_all_slided)

    masked_dt_all = dt_all[mask]

    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(masked_dt_all)

    # np.correlate(M, N): Mが0パディングされる側、Nがスライドする側
    (
        corr_with_real_and_calc,
        estimated_delay_with_real_and_calc,
    ) = calc_delay(masked_q_all, masked_calc_q_all)
    print(f"ずれ時間（実測値と計算値）: {estimated_delay_with_real_and_calc}[s]")

    (
        corr_with_real_and_calc_slided,
        estimated_delay_with_real_and_calc_slided,
    ) = calc_delay(masked_q_all, masked_calc_q_all_slided)
    print(f"ずれ時間（実測値と計算値（ずらし有り））: {estimated_delay_with_real_and_calc_slided}[s]")

    # ずらしありの計算値列を左から右へスライドさせていく
    (
        corr_with_calc_and_calc_slided,
        estimated_delay_with_calc_and_calc_slided,
    ) = calc_delay(masked_calc_q_all, masked_calc_q_all_slided)
    print(f"ずれ時間(計算値と計算値（ずらし有り）): {estimated_delay_with_calc_and_calc_slided}[s]")

    # 実測値と計算値
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
    axes[0].set_title(f"ずれ時間: {estimated_delay_with_real_and_calc}[s]")
    axes[0].set_xlabel("時刻")
    axes[0].set_ylabel("日射量[kW/m^2]")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[0].legend()

    # 実測値と計算値（ずらし有り）
    axes[1].plot(
        unified_dates,
        masked_q_all,
        label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[1].plot(
        unified_dates,
        masked_calc_q_all_slided,
        label=f"計算値(ずらし有り): {dt_all[0].strftime('%Y-%m-%d')}",
        linestyle="dashed",
        color=colorlist[1],
    )
    axes[1].set_title(f"ずれ時間: {estimated_delay_with_real_and_calc_slided}[s]")
    axes[1].set_xlabel("時刻")
    axes[1].set_ylabel("日射量[kW/m^2]")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[1].legend()

    # 計算値同と計算値（ずらし有り）
    axes[2].plot(
        unified_dates,
        masked_calc_q_all,
        label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[2].plot(
        unified_dates,
        masked_calc_q_all_slided,
        label=f"計算値(ずらし有り): {dt_all[0].strftime('%Y-%m-%d')}",
        linestyle="dashed",
        color=colorlist[1],
    )
    # axes[2].plot(
    #     unified_dates,
    #     np.roll(masked_calc_q_all_slided, args.slide_seconds),
    #     label=f"計算値(ずらし有りをロール): {dt_all[0].strftime('%Y-%m-%d')}",
    #     linestyle="dashdot",
    #     color=colorlist[2],
    # )
    axes[2].set_title(f"ずれ時間: {estimated_delay_with_calc_and_calc_slided}[s]")
    axes[2].set_xlabel("時刻")
    axes[2].set_ylabel("日射量[kW/m^2]")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[2].legend()

    lags = np.concatenate(
        [
            np.arange(-1 * len(masked_calc_q_all) + 1, 0, 1),
            np.arange(0, len(masked_calc_q_all), 1),
        ],
        0,
    )
    axes[3].plot(
        lags,
        corr_with_real_and_calc,
        label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[3].set_xlabel("ラグ")
    axes[3].set_ylabel("相互相関")
    axes[3].legend()

    lags = np.concatenate(
        [
            np.arange(-1 * len(masked_calc_q_all_slided) + 1, 0, 1),
            np.arange(0, len(masked_calc_q_all_slided), 1),
        ],
        0,
    )
    axes[4].plot(
        lags,
        corr_with_real_and_calc_slided,
        label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[4].set_xlabel("ラグ")
    axes[4].set_ylabel("相互相関")
    axes[4].legend()

    lags = np.concatenate(
        [
            np.arange(-1 * len(masked_calc_q_all_slided) + 1, 0, 1),
            np.arange(0, len(masked_calc_q_all_slided), 1),
        ],
        0,
    )
    axes[5].plot(
        lags,
        corr_with_calc_and_calc_slided,
        label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[5].set_xlabel("ラグ")
    axes[5].set_ylabel("相互相関")
    axes[5].legend()

    plt.show()
