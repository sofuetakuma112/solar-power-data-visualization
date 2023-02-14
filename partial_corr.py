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
    dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
    dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

    # 昇順で並んでいるかテストする
    sort_indexes = np.argsort(dt_all)

    print(
        f"np.allclose(sort_indexes, np.arange(0, dt_all.size, 1)): {np.allclose(sort_indexes, np.arange(0, dt_all.size, 1))}"
    )

    q = Q()  # インスタンス作成時にDBへのコネクションを初期化
    calced_q_all = q.calc_qs_kw_v2(
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=22,
        surface_azimuth=185,
        model="isotropic",
    )

    SECONDS_FOR_SLIDE = 6 * 60 * 60
    calced_q_all_slided = q.calc_qs_kw_v2(
        # dt_all + datetime.timedelta(seconds=870),
        dt_all + datetime.timedelta(seconds=SECONDS_FOR_SLIDE),
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=22,
        surface_azimuth=185,
        model="isotropic",
    )
    # calced_q_all = np.vectorize(calc_q_kw)(dt_all)

    axes = [plt.subplots()[1] for _ in range(4)]

    # 08:30 ~
    # mask = dt_all > datetime.datetime(int(year), int(month), int(date), 8, 30, 0)

    # 08:30 ~ 15:30
    # mask = (
    #     (datetime.datetime(int(year), int(month), int(date), 15, 30, 0)) > dt_all
    # ) & (dt_all > datetime.datetime(int(year), int(month), int(date), 8, 30, 0))

    # 07:20 ~ 17:10
    mask = (
        (datetime.datetime(int(year), int(month), int(date), 17, 10, 0)) > dt_all
    ) & (dt_all > datetime.datetime(int(year), int(month), int(date), 7, 20, 0))

    # 1日全て
    # mask = (
    #     (
    #         datetime.datetime(int(year), int(month), int(date), 0, 0, 0)
    #         + datetime.timedelta(days=1)
    #     )
    #     > dt_all
    # ) & (dt_all > datetime.datetime(int(year), int(month), int(date), 0, 0, 0))

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

    # TODO: 実測値と計算値（ずらし有り）で相互相関を取って、相互相関の秒数の検知精度を調べる

    # np.correlate(M, N): Mが0パディングされる側、Nがスライドする側
    corr1 = np.correlate(masked_q_all, masked_calc_q_all, "full")
    estimated_delay1 = corr1.argmax() - (len(masked_calc_q_all) - 1)
    print(f"ずれ時間: {estimated_delay1}[s]")

    # ずらしありの計算値列を左から右へスライドさせていく
    corr2 = np.correlate(masked_calc_q_all, masked_calc_q_all_slided, "full")
    estimated_delay2 = corr2.argmax() - (len(masked_calc_q_all_slided) - 1)
    print(f"ずれ時間(calc_q同士): {estimated_delay2}[s]")

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
    axes[0].set_title(f"ずれ時間: {estimated_delay1}[s]")
    axes[0].set_xlabel("時刻")
    axes[0].set_ylabel("日射量[kW/m^2]")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[0].legend()

    # 計算値同士
    axes[1].plot(
        unified_dates,
        masked_calc_q_all,
        label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[1].plot(
        unified_dates,
        masked_calc_q_all_slided,
        label=f"計算値(ずらし有り): {dt_all[0].strftime('%Y-%m-%d')}",
        linestyle="dashed",
        color=colorlist[1],
    )
    # axes[1].plot(
    #     unified_dates,
    #     np.roll(masked_calc_q_all_slided, SECONDS_FOR_SLIDE),
    #     label=f"計算値(ずらし有りをロール): {dt_all[0].strftime('%Y-%m-%d')}",
    #     linestyle="dashdot",
    #     color=colorlist[2],
    # )
    axes[1].set_title(f"ずれ時間: {estimated_delay2}[s]")
    axes[1].set_xlabel("時刻")
    axes[1].set_ylabel("日射量[kW/m^2]")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[1].legend()

    lags = np.concatenate(
        [
            np.arange(-1 * len(masked_calc_q_all) + 1, 0, 1),
            np.arange(0, len(masked_calc_q_all), 1),
        ],
        0,
    )
    axes[2].plot(
        lags,
        corr1,
        label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[2].set_xlabel("ラグ")
    axes[2].set_ylabel("相互相関")
    axes[2].legend()

    lags = np.concatenate(
        [
            np.arange(-1 * len(masked_calc_q_all_slided) + 1, 0, 1),
            np.arange(0, len(masked_calc_q_all_slided), 1),
        ],
        0,
    )
    axes[3].plot(
        lags,
        corr2,
        label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[3].set_xlabel("ラグ")
    axes[3].set_ylabel("相互相関")
    axes[3].legend()

    plt.show()
