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
from utils.init_matplotlib import init_rcParams, figsize_px_to_inch

FONT_SIZE = 14


def advance_or_delay(seconds):
    if np.sign(seconds) == 1:
        return f"{seconds}[s]進めている"
    elif np.sign(seconds) == -1:
        return f"{seconds}[s]遅らせている"
    else:
        return ""


def min0_max1(data):
    # 最小値と最大値を計算
    min_value = np.min(data)
    max_value = np.max(data)

    # 最小0最大1に変換
    return (data - min_value) / (max_value - min_value)


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


def correlate_full(x, y):
    n = x.size
    m = y.size
    result = np.array([0] * (n + m - 1))
    for i in range(n):
        for j in range(m):
            result[i + j] += x[i] * y[j]
    return result


def update_row_and_column_index(crr_row_idx, crr_column_idx, rows, columns):
    if crr_column_idx + 1 == columns:
        if crr_row_idx + 1 == rows:
            return -1, -1
        return [crr_row_idx + 1, 0]

    return [crr_row_idx, crr_column_idx + 1]


# > python3 partial_corr.py -dt 2022/06/02 -slide_seconds 1000 -surface_tilt 22 -surface_azimuth 179 -mask_from 07:20 -mask_to 17:10
# > python3 selective_corr.py -dt 2022/06/02 -slide_seconds 0 -surface_tilt 22 -surface_azimuth 179
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
    parser.add_argument("-threshold_q", type=float, default=0.2)
    parser.add_argument("-h_rac", action="store_true")  # 実測値と計算値
    parser.add_argument(
        "-h_racs", action="store_true"
    )  # 実測値と計算値（ずらし）、real and calc slide
    parser.add_argument(
        "-h_cacs", action="store_true"
    )  # 計算値と計算値（ずらし）、calc and calc slide
    parser.add_argument("-h_cc", action="store_true")  # 相互相関、cross correlation
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

    # 実測値を平滑化スプラインで滑らかにする
    x = np.arange(0, dt_all.size, 1)
    model = get_natural_cubic_spline_model(
        x, q_all, minval=min(x), maxval=max(x), n_knots=18
    )
    q_all_spline = model.predict(x)

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
        masked_q_all = q_all[mask]
        masked_q_all_spline = q_all_spline[mask]

        masked_dt_all = dt_all[mask]
    elif args.masking_strategy == "replace":
        inverted_mask = np.logical_not(mask)
        np.putmask(
            q_all_spline, inverted_mask, (q_all_spline * 0) + np.min(q_all_spline[mask])
        )
        np.putmask(q_all, inverted_mask, (q_all * 0) + np.min(q_all[mask]))

        masked_q_all = q_all
        masked_q_all_spline = q_all_spline

        masked_dt_all = dt_all
    elif args.masking_strategy == "replace_zero":
        inverted_mask = np.logical_not(mask)
        np.putmask(q_all_spline, inverted_mask, q_all_spline * 0)
        np.putmask(q_all, inverted_mask, q_all * 0)

        masked_q_all = q_all
        masked_q_all_spline = q_all_spline

        masked_dt_all = dt_all
    else:
        raise ValueError("masking_strategyの値が不正")

    calc_q_all_slided = np.roll(calc_q_all, -args.slide_seconds)

    if args.threshold_q != 0.0:
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

        # threshold_q_masked_q_all = q_all
        # threshold_q_masked_dt_all = dt_all

        # threshold_q_mask_from, threshold_q_mask_toの外側をすべての0にした
        # threshold_q_masked_q_all_and_replaced_zero = np.copy(threshold_q_masked_q_all)

        # threshold_q_mask_from ~ threshold_q_mask_toのすべての点で「実測値列 - 指定したq」を求めて、マイナスになった箇所は0にする
        # threshold_q_masked_q_all_and_subed = threshold_q_masked_q_all - args.threshold_q
        # threshold_q_masked_q_all_and_subed[threshold_q_masked_q_all_and_subed < 0] = 0

    # 標準化
    if args.normalize:
        masked_q_all = normalize(masked_q_all)
        masked_q_all_spline = normalize(masked_q_all_spline)
        calc_q_all = normalize(calc_q_all)
        calc_q_all_slided = normalize(calc_q_all_slided)

    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(masked_dt_all)

    # np.correlate(M, N): Mが0パディングされる側、Nがスライドする側
    (
        corr_with_real_and_calc,
        estimated_delay_with_real_and_calc,
    ) = calc_delay(calc_q_all, masked_q_all)
    print(f"ずれ時間（実測値と計算値）: {estimated_delay_with_real_and_calc}[s]")

    if args.slide_seconds != 0:
        (
            corr_with_real_and_calc_slided,
            estimated_delay_with_real_and_calc_slided,
        ) = calc_delay(calc_q_all_slided, masked_q_all)
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

    if args.h_cc:
        total_figure_count = 3
        count_of_fig_per_unit = 1
    else:
        total_figure_count = 6
        count_of_fig_per_unit = 2

    if args.h_rac:
        total_figure_count -= count_of_fig_per_unit
    if args.h_racs:
        total_figure_count -= count_of_fig_per_unit
    if args.h_cacs:
        total_figure_count -= count_of_fig_per_unit

    if total_figure_count == 0:
        exit()

    rows = int(count_of_fig_per_unit)
    columns = int(total_figure_count / count_of_fig_per_unit)

    print(f"rows: {rows}, columns: {columns}")

    if rows > columns:
        # rowsとcolumsをひっくり返す
        rows, columns = columns, rows

    figsize_inch = figsize_px_to_inch(np.array([1280, 720]))
    plt.rcParams = init_rcParams(plt.rcParams, FONT_SIZE, figsize_inch)

    fig, axes = plt.subplots(rows, columns)
    fig.set_constrained_layout(True)

    if rows == 1 and columns == 1:
        # HACK: 参照エラーを回避するため
        axes = np.append(axes, "適当な値")

    if rows == 1:
        axes = axes.reshape(1, -1)

    crr_row_idx = 0
    crr_column_idx = 0

    span = f"{mask_from.strftime('%Y-%m-%d %H:%M:%S')}〜{mask_to.strftime('%Y-%m-%d %H:%M:%S')}"

    if not args.h_rac:
        # 実測値と計算値
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            masked_q_all,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            calc_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"実測値と計算値\nずれ時間: {estimated_delay_with_real_and_calc}[s]\n{span}",
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("時刻")
        axes[crr_row_idx, crr_column_idx].set_ylabel("日射量 [kW/m$^2$]")
        axes[crr_row_idx, crr_column_idx].xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_racs:
        # 実測値と計算値（ずらし有り）
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            masked_q_all,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            calc_q_all_slided,
            label=f"計算値({advance_or_delay(args.slide_seconds)}): {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"実測値と計算値（{advance_or_delay(args.slide_seconds)}）\nずれ時間: {estimated_delay_with_real_and_calc_slided}[s]\n{span}",
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("時刻")
        axes[crr_row_idx, crr_column_idx].set_ylabel("日射量 [kW/m$^2$]")
        axes[crr_row_idx, crr_column_idx].xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_cacs:
        # 計算値同と計算値（ずらし有り）
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            calc_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            calc_q_all_slided,
            label=f"計算値({advance_or_delay(args.slide_seconds)}): {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"計算値と計算値（{advance_or_delay(args.slide_seconds)}）\nずれ時間: {estimated_delay_with_calc_and_calc_slided}[s]\n{span}",
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("時刻")
        axes[crr_row_idx, crr_column_idx].set_ylabel("日射量 [kW/m$^2$]")
        axes[crr_row_idx, crr_column_idx].xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_rac and not args.h_cc:
        lags = np.concatenate(
            [
                np.arange(-1 * len(calc_q_all) + 1, 0, 1),
                np.arange(0, len(calc_q_all), 1),
            ],
            0,
        )
        axes[crr_row_idx, crr_column_idx].plot(
            lags,
            corr_with_real_and_calc,
            label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].set_title(f"実測値と計算値")
        axes[crr_row_idx, crr_column_idx].set_xlabel("ラグ")
        axes[crr_row_idx, crr_column_idx].set_ylabel("相互相関")
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_racs and not args.h_cc:
        lags = np.concatenate(
            [
                np.arange(-1 * len(calc_q_all_slided) + 1, 0, 1),
                np.arange(0, len(calc_q_all_slided), 1),
            ],
            0,
        )
        axes[crr_row_idx, crr_column_idx].plot(
            lags,
            corr_with_real_and_calc_slided,
            label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"実測値と計算値（{advance_or_delay(args.slide_seconds)}）"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("ラグ")
        axes[crr_row_idx, crr_column_idx].set_ylabel("相互相関")
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_cacs and not args.h_cc:
        lags = np.concatenate(
            [
                np.arange(-1 * len(calc_q_all_slided) + 1, 0, 1),
                np.arange(0, len(calc_q_all_slided), 1),
            ],
            0,
        )
        axes[crr_row_idx, crr_column_idx].plot(
            lags,
            corr_with_calc_and_calc_slided,
            label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"計算値と計算値（{advance_or_delay(args.slide_seconds)}）"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("ラグ")
        axes[crr_row_idx, crr_column_idx].set_ylabel("相互相関")
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    plt.show()
