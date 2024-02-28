import datetime
import matplotlib.pyplot as plt

import matplotlib_fontja
from utils.corr import calc_delay
from utils.date import mask_from_into_dt, mask_to_into_dt
from utils.es.load import load_q_and_dt_for_period
import argparse
import numpy as np
from utils.q import Q
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates

from utils.init_matplotlib import init_rcParams, figsize_px_to_inch

FONT_SIZE = 20


def advance_or_delay(seconds):
    if np.sign(seconds) == 1:
        return f"{seconds}[s]進めている"
    elif np.sign(seconds) == -1:
        return f"{seconds}[s]遅らせている"
    else:
        return ""


# > python3 selective_corr.py -dt 2022/06/02 -slide_seconds 0 -surface_tilt 22 -surface_azimuth 179 -threshold_q 0.2 -show_preprocessing_data -show_threshold_q
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    parser.add_argument(
        "-slide_seconds", type=int, default=0
    )  # 正の値の場合は左にスライド、負の場合は右にスライド
    parser.add_argument("-mask_from", type=str, default="00:00")
    parser.add_argument("-mask_to", type=str, default="24:00")
    parser.add_argument("-masking_strategy", type=str, default="drop")
    parser.add_argument(
        "-model", type=str, default="isotropic"
    )  # 日射量推定に使用するモデル
    parser.add_argument("-surface_tilt", type=int, default=22) # 太陽光パネルの傾き
    parser.add_argument("-surface_azimuth", type=float, default=185.0) # 太陽光パネルの向き（360°）
    parser.add_argument("-threshold_q", type=float, default=0.0) # 前処理のしきい値
    parser.add_argument("-show_threshold_q", action="store_true") # 前処理のしきい値をグラフに表示するか
    parser.add_argument("-show_preprocessing_data", action="store_true") # 前処理前の実測値を表示するか
    args = parser.parse_args()

    if args.slide_seconds == 0:
        args.h_racs = True
        args.h_cacs = True

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

    q = Q()  # インスタンス作成時にDBへのコネクションを初期化
    calc_q_all = q.calc_qs_kw_v2(
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model=args.model,
    )

    print(f"真のズレ時間: {args.slide_seconds}[s]")

    mask_from = mask_from_into_dt(args.mask_from, year, month, day)
    mask_to = mask_to_into_dt(args.mask_to, year, month, day)

    print(f"{mask_from} 〜 {mask_to}")

    mask = (mask_from <= dt_all) & (dt_all < mask_to)

    # マスク処理
    if args.masking_strategy == "drop":
        q_all = q_all[mask]

        dt_all = dt_all[mask]
    else:
        raise ValueError("masking_strategyの値が不正")

    calc_q_all_slided = np.roll(calc_q_all, -args.slide_seconds)

    if args.threshold_q != 0.0:
        preprocessing_q_all = np.copy(q_all)

        # しきい値のQでフィルタリング処理
        # 1. 12時の左側と右側でそれぞれ1点ずつ指定したqの値に最も近い点のタイムスタンプを探す
        diffs_from_noon = dt_all - datetime.datetime(
            int(from_dt.year), int(from_dt.month), int(from_dt.day), 12, 0, 0
        )
        noon_idx = np.argmin(
            np.vectorize(lambda diff_delta: np.abs(diff_delta.total_seconds()))(
                diffs_from_noon
            )
        )

        print(f"dt_all[noon_idx]: {dt_all[noon_idx]}")

        # 2.a 午前で実測値が指定した値に最も近いときのtimestampを取得する
        q_all_copy = np.copy(q_all)
        left_timestamp_idx = np.argmin(np.abs(q_all_copy[:noon_idx] - args.threshold_q))
        threshold_q_mask_from = dt_all[left_timestamp_idx]
        # 2.b 午後で実測値が指定した値に最も近いときのtimestampを取得する
        q_all_copy = np.copy(q_all)
        right_timestamp_idx = (
            np.argmin(np.abs(q_all_copy[noon_idx:] - args.threshold_q)) + noon_idx
        )
        threshold_q_mask_to = dt_all[right_timestamp_idx]

        # 3. 実測値を取得したタイムスタンプでマスキングする
        print(f"{threshold_q_mask_from} 〜 {threshold_q_mask_to}")

        mask = (threshold_q_mask_from <= dt_all) & (dt_all < threshold_q_mask_to)

        # 実測値のthreshold_q_mask_from ~ threshold_q_mask_to以外を0に置き換える
        inverted_mask = np.logical_not(mask)
        np.putmask(q_all, inverted_mask, q_all * 0)

    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(dt_all)

    # np.correlate(M, N): Mが0パディングされる側、Nがスライドする側
    (
        corr_with_real_and_calc,
        estimated_delay_with_real_and_calc,
    ) = calc_delay(calc_q_all, q_all)
    print(f"ずれ時間（実測値と計算値）: {estimated_delay_with_real_and_calc}[s]")

    if args.slide_seconds != 0:
        (
            corr_with_real_and_calc_slided,
            estimated_delay_with_real_and_calc_slided,
        ) = calc_delay(calc_q_all_slided, q_all)
        print(
            f"ずれ時間（実測値と計算値（{advance_or_delay(args.slide_seconds)}））: {estimated_delay_with_real_and_calc_slided}[s]"
        )

    if args.slide_seconds != 0:
        # ずらしありの計算値列を左から右へスライドさせていく
        (
            corr_with_calc_and_calc_slided,
            estimated_delay_with_calc_and_calc_slided,
        ) = calc_delay(calc_q_all_slided, calc_q_all)
        print(
            f"ずれ時間(計算値と計算値（{advance_or_delay(args.slide_seconds)}）): {estimated_delay_with_calc_and_calc_slided}[s]"
        )

    figsize_inch = figsize_px_to_inch(np.array([1280, 720]))
    plt.rcParams = init_rcParams(plt.rcParams, FONT_SIZE, figsize_inch)

    fig, ax = plt.subplots()
    fig.set_constrained_layout(True)

    span = f"{mask_from.strftime('%Y-%m-%d %H:%M:%S')}〜{mask_to.strftime('%Y-%m-%d %H:%M:%S')}"

    # 実測値と計算値
    if args.show_preprocessing_data:
        ax.plot(
            unified_dates,
            preprocessing_q_all,
            label=f"実測値",
            color=colorlist[2],
        )
    ax.plot(
        unified_dates,
        q_all,
        label=f"{"前処理後の" if args.threshold_q != 0.0 else ""}実測値",
        color=colorlist[0],
    )
    ax.plot(
        unified_dates,
        calc_q_all,
        label=f"計算値",
        linestyle="dashed",
        color=colorlist[1],
    )
    if args.show_threshold_q:
        ax.axhline(y=args.threshold_q, linestyle='--', color=colorlist[3], label="前処理用のしきい値")
    ax.set_xlabel("時刻")
    ax.set_ylabel("日射量 [kW/m$^2$]")
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )
    ax.legend()

    plt.show()
