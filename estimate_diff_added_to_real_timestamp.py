import datetime
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.corr import calc_delay
from utils.date import mask_from_into_dt, mask_to_into_dt
from utils.es.load import load_q_and_dt_for_period
import argparse
import numpy as np
from utils.q import Q
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from scipy import interpolate

from utils.spline_model import get_natural_cubic_spline_model

import multiprocessing

# 実測値のズレたタイムスタンプ列をマスクする
def masking(calced_q_all):
    mask_from = mask_from_into_dt(args.mask_from, year, month, day)
    mask_to = mask_to_into_dt(args.mask_to, year, month, day)

    mask = (mask_from <= dt_all) & (dt_all < mask_to)

    # マスク処理
    if args.masking_strategy == "drop":
        masked_q_all = q_all[mask]
        masked_calc_q_all = calced_q_all[mask]

        masked_dt_all = dt_all[mask]
    elif args.masking_strategy == "replace":
        inverted_mask = np.logical_not(mask)
        np.putmask(q_all, inverted_mask, (q_all * 0) + np.min(q_all[mask]))
        np.putmask(
            calced_q_all,
            inverted_mask,
            (calced_q_all * 0) + np.min(calced_q_all[mask]),
        )

        masked_q_all = q_all
        masked_calc_q_all = calced_q_all

        masked_dt_all = dt_all
    elif args.masking_strategy == "replace_zero":
        inverted_mask = np.logical_not(mask)
        np.putmask(q_all, inverted_mask, q_all * 0)
        np.putmask(calced_q_all, inverted_mask, calced_q_all * 0)

        masked_q_all = q_all
        masked_calc_q_all = calced_q_all

        masked_dt_all = dt_all
    else:
        raise ValueError("masking_strategyの値が不正")

    return masked_q_all, masked_calc_q_all, masked_dt_all


def thread_process(lag):
    print(f"current lag: {lag}[s]")
    dt_all_with_lag = dt_all + datetime.timedelta(seconds=int(lag))

    calced_q_all = q.calc_qs_kw_v2(
        dt_all_with_lag,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model=args.model,
    )

    masked_q_all, masked_calc_q_all, _ = masking(calced_q_all)

    sum_of_products = np.dot(masked_calc_q_all, masked_q_all)

    return np.array([sum_of_products.sum(), lag])

# > python3 estimate_diff_added_to_real_timestamp.py -dt 2022/04/08 -surface_tilt 28 -surface_azimuth 178.28 -h_rpacs -h_cacs -h_cc -mask_from 07:20 -mask_to 17:10 -slide_seconds 0 -masking_strategy replace_zero
# > python3 estimate_diff_added_to_real_timestamp.py -dt 2022/06/02 -surface_tilt 28 -surface_azimuth 178.28 -h_rpacs -h_cacs -h_cc -mask_from 06:35 -mask_to 17:35 -slide_seconds 0 -masking_strategy replace_zero
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    parser.add_argument(
        "-slide_seconds", type=int, default=0
    )  # 正の値の場合は左にスライド、負の場合は右にスライド
    parser.add_argument("-mask_from", type=str, default="00:00")
    parser.add_argument("-mask_to", type=str, default="24:00")
    parser.add_argument("-masking_strategy", type=str, default="drop")
    parser.add_argument("-normalize", action="store_true")
    parser.add_argument(
        "-model", type=str, default="isotropic"
    )  # 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    parser.add_argument("-surface_tilt", type=int, default=22)
    parser.add_argument("-surface_azimuth", type=float, default=185.0)
    parser.add_argument("-h_rac", action="store_true")  # 実測値と計算値
    parser.add_argument(
        "-h_racs", action="store_true"
    )  # 実測値と計算値（ずらし）、real and calc slide
    parser.add_argument(
        "-h_rpacs", action="store_true"
    )  # 実測値（スプライン）と計算値（ずらし）、real spline and calc slide
    parser.add_argument(
        "-h_cacs", action="store_true"
    )  # 計算値と計算値（ずらし）、calc and calc slide
    parser.add_argument("-h_cc", action="store_true")  # 相互相関、cross correlation
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

    # 昇順で並んでいるかテストする
    sort_indexes = np.argsort(dt_all)
    if not np.allclose(sort_indexes, np.arange(0, dt_all.size, 1)):
        raise ValueError("dt_allが時系列順で並んでいない")

    # 実測値のタイムスタンプを意図的にずらす
    dt_all += datetime.timedelta(seconds=args.slide_seconds)

    print(f"真のズレ時間: {args.slide_seconds}[s]")
    # print(f"dt_all[0], dt_all[-1]: {dt_all[0]}, {dt_all[-1]}")

    q = Q()

    # -args.slide_seconds ~ args.slide_secondsの間で1s刻みで相互相関を求めて最大を取ったときのdt_allに加えたズレ時間を求める
    results = np.empty((0, 2))
    seconds = 300
    lags = np.arange(-seconds, seconds, 1)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        returns = p.map(
            thread_process,
            lags,
        )
        results = np.array(returns).reshape(-1, 2)

    max_row = results[results[:, 0].argmax()]

    axes = [plt.subplots()[1] for _ in range(2)]

    axes[0].plot(
        results[:, 1],
        results[:, 0],
        color=colorlist[0],
    )

    axes[0].set_title(f"加えたラグに対応する積和")
    axes[0].set_xlabel("加えたラグ")
    axes[0].set_ylabel("積和")

    ## 積和が最大のときのlagだけdt_allをずらしてから求めた計算値と元の実測値をプロットする
    best_lag = max_row[1]
    dt_all_with_lag = dt_all + datetime.timedelta(seconds=int(best_lag))

    calced_q_all = q.calc_qs_kw_v2(
        dt_all_with_lag,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model=args.model,
    )

    masked_q_all, masked_calc_q_all, masked_dt_all = masking(calced_q_all)

    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(masked_dt_all)

    axes[1].plot(
        unified_dates,
        masked_q_all,
        label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
        color=colorlist[0],
    )
    axes[1].plot(
        unified_dates,
        masked_calc_q_all,
        label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
        linestyle="dashed",
        color=colorlist[1],
    )
    axes[1].set_title(f"計算値を求める際にdt_allに加えたlag: {best_lag}[s]")
    axes[1].set_xlabel("時刻")
    axes[1].set_ylabel("日射量[kW/m^2]")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].legend()

    plt.show()
    plt.clf()
    plt.close()

    plt.show()
