import datetime
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import argparse
import numpy as np
from utils.q import Q
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from scipy import interpolate

from utils.spline_model import get_natural_cubic_spline_model


def calc_delay(a, b):
    corr = np.correlate(a, b, "full")
    return [corr, corr.argmax() - (len(b) - 1)]


def advance_or_delay(seconds):
    if np.sign(seconds) == 1:
        return "進めている"
    elif np.sign(seconds) == -1:
        return "遅らせている"
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


# > python3 partial_corr.py -dt 2022/04/08 -slide_seconds 1000 -mask_from 07:20 -mask_to 17:10
# > python3 partial_corr.py -dt 2022/04/08 -slide_seconds 10
# > python3 partial_corr.py -dt 2022/04/08 -surface_tilt 26 -surface_azimuth 180.5 -h_racs -h_rpacs -h_cacs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    parser.add_argument("-slide_seconds", type=int, default=0)
    parser.add_argument("-mask_from", type=str, default="00:00")
    parser.add_argument("-mask_to", type=str, default="24:00")
    parser.add_argument("-masking_strategy", type=str, default="drop")
    parser.add_argument("-normalize", action="store_true")
    parser.add_argument(
        "-model", type=str, default="isotropic"
    )  # 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    parser.add_argument("-surface_tilt", type=int, default=22)
    parser.add_argument("-surface_azimuth", type=float, default=185.0)
    parser.add_argument("-h_rac", action="store_true") # 実測値と計算値
    parser.add_argument("-h_racs", action="store_true") # 実測値と計算値（ずらし）
    parser.add_argument("-h_rpacs", action="store_true") # 実測値（スプライン）と計算値（ずらし）
    parser.add_argument("-h_cacs", action="store_true") # 計算値と計算値（ずらし）
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

    # 実測値を平滑化スプラインで滑らかにする
    x = np.arange(0, dt_all.size, 1)
    model = get_natural_cubic_spline_model(
        x, q_all, minval=min(x), maxval=max(x), n_knots=18
    )
    q_all_spline = model.predict(x)

    q = Q()  # インスタンス作成時にDBへのコネクションを初期化
    calced_q_all = q.calc_qs_kw_v2(
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model=args.model,
    )

    print(f"真のズレ時間: {args.slide_seconds}[s]")
    calced_q_all_slided = q.calc_qs_kw_v2(
        dt_all + datetime.timedelta(seconds=args.slide_seconds),
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model="isotropic",
    )

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

    print(f"{mask_from} 〜 {mask_to}")

    # 1日全て
    mask = (mask_from <= dt_all) & (dt_all < mask_to)

    # マスク処理
    if args.masking_strategy == "drop":
        masked_q_all = q_all[mask]
        masked_q_all_spline = q_all_spline[mask]
        masked_calc_q_all = calced_q_all[mask]
        masked_calc_q_all_slided = calced_q_all_slided[mask]

        masked_dt_all = dt_all[mask]
    elif args.masking_strategy == "replace":
        inverted_mask = np.logical_not(mask)
        np.putmask(
            q_all_spline, inverted_mask, (q_all_spline * 0) + np.min(q_all_spline[mask])
        )
        np.putmask(q_all, inverted_mask, (q_all * 0) + np.min(q_all[mask]))
        np.putmask(
            calced_q_all, inverted_mask, (calced_q_all * 0) + np.min(calced_q_all[mask])
        )
        np.putmask(
            calced_q_all_slided,
            inverted_mask,
            (calced_q_all_slided * 0) + np.min(calced_q_all_slided[mask]),
        )

        masked_q_all = q_all
        masked_q_all_spline = q_all_spline
        masked_calc_q_all = calced_q_all
        masked_calc_q_all_slided = calced_q_all_slided

        masked_dt_all = dt_all
    else:
        raise ValueError("masking_strategyの値が不正")

    # 標準化
    if args.normalize:
        masked_q_all = normalize(masked_q_all)
        masked_q_all_spline = normalize(masked_q_all_spline)
        masked_calc_q_all = normalize(masked_calc_q_all)
        masked_calc_q_all_slided = normalize(masked_calc_q_all_slided)

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
    print(
        f"ずれ時間（実測値と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}））: {estimated_delay_with_real_and_calc_slided}[s]"
    )

    (
        corr_with_real_spline_and_calc_slided,
        estimated_delay_with_real_spline_and_calc_slided,
    ) = calc_delay(masked_q_all_spline, masked_calc_q_all_slided)
    print(
        f"ずれ時間（実測値（スプライン）と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}））: {estimated_delay_with_real_spline_and_calc_slided}[s]"
    )

    # ずらしありの計算値列を左から右へスライドさせていく
    (
        corr_with_calc_and_calc_slided,
        estimated_delay_with_calc_and_calc_slided,
    ) = calc_delay(masked_calc_q_all, masked_calc_q_all_slided)
    print(
        f"ずれ時間(計算値と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}）): {estimated_delay_with_calc_and_calc_slided}[s]"
    )

    total_figure_count = 8

    if args.h_rac:
        total_figure_count -= 2
    if args.h_racs:
        total_figure_count -= 2
    if args.h_rpacs:
        total_figure_count -= 2
    if args.h_cacs:
        total_figure_count -= 2

    if total_figure_count == 0:
        exit()

    # 8, 6, 4, 2のケースがある
    rows = 2
    columns = int(total_figure_count / 2)

    if rows > columns:
        # rowsとcolumsをひっくり返す
        rows, columns = columns, rows

    print(f"rows: {rows} columns: {columns}")

    fig, axes = plt.subplots(rows, columns)
    fig.set_constrained_layout(True)

    if rows == 1:
        axes = axes.reshape(1, -1)

    crr_row_idx = 0
    crr_column_idx = 0

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
            masked_calc_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"実測値と計算値\nずれ時間: {estimated_delay_with_real_and_calc}[s]"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("時刻")
        axes[crr_row_idx, crr_column_idx].set_ylabel("日射量[kW/m^2]")
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
            masked_calc_q_all_slided,
            label=f"計算値({args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}): {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"実測値と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}）\nずれ時間: {estimated_delay_with_real_and_calc_slided}[s]"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("時刻")
        axes[crr_row_idx, crr_column_idx].set_ylabel("日射量[kW/m^2]")
        axes[crr_row_idx, crr_column_idx].xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_rpacs:
        # 実測値（スプライン）と計算値（ずらし有り）
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            masked_q_all_spline,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            masked_calc_q_all_slided,
            label=f"計算値({args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}): {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"実測値（スプライン）と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}）\nずれ時間: {estimated_delay_with_real_spline_and_calc_slided}[s]"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("時刻")
        axes[crr_row_idx, crr_column_idx].set_ylabel("日射量[kW/m^2]")
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
            masked_calc_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].plot(
            unified_dates,
            masked_calc_q_all_slided,
            label=f"計算値({args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}): {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        # axes[crr_row_idx, crr_column_idx].plot(
        #     unified_dates,
        #     np.roll(masked_calc_q_all_slided, args.slide_seconds),
        #     label=f"計算値(ずらし有りをロール): {dt_all[0].strftime('%Y-%m-%d')}",
        #     linestyle="dashdot",
        #     color=colorlist[2],
        # )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"計算値と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}）\nずれ時間: {estimated_delay_with_calc_and_calc_slided}[s]"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("時刻")
        axes[crr_row_idx, crr_column_idx].set_ylabel("日射量[kW/m^2]")
        axes[crr_row_idx, crr_column_idx].xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_rac:
        lags = np.concatenate(
            [
                np.arange(-1 * len(masked_calc_q_all) + 1, 0, 1),
                np.arange(0, len(masked_calc_q_all), 1),
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

    if not args.h_racs:
        lags = np.concatenate(
            [
                np.arange(-1 * len(masked_calc_q_all_slided) + 1, 0, 1),
                np.arange(0, len(masked_calc_q_all_slided), 1),
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
            f"実測値と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}）"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("ラグ")
        axes[crr_row_idx, crr_column_idx].set_ylabel("相互相関")
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_rpacs:
        lags = np.concatenate(
            [
                np.arange(-1 * len(masked_calc_q_all_slided) + 1, 0, 1),
                np.arange(0, len(masked_calc_q_all_slided), 1),
            ],
            0,
        )
        axes[crr_row_idx, crr_column_idx].plot(
            lags,
            corr_with_real_spline_and_calc_slided,
            label=f"相互相関: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[crr_row_idx, crr_column_idx].set_title(
            f"実測値（スプライン）と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}）"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("ラグ")
        axes[crr_row_idx, crr_column_idx].set_ylabel("相互相関")
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    if not args.h_cacs:
        lags = np.concatenate(
            [
                np.arange(-1 * len(masked_calc_q_all_slided) + 1, 0, 1),
                np.arange(0, len(masked_calc_q_all_slided), 1),
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
            f"計算値と計算値（{args.slide_seconds}[s]{advance_or_delay(args.slide_seconds)}）"
        )
        axes[crr_row_idx, crr_column_idx].set_xlabel("ラグ")
        axes[crr_row_idx, crr_column_idx].set_ylabel("相互相関")
        axes[crr_row_idx, crr_column_idx].legend()

        crr_row_idx, crr_column_idx = update_row_and_column_index(
            crr_row_idx, crr_column_idx, rows, columns
        )

    plt.show()
