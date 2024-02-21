import argparse
import datetime
from matplotlib import dates
import matplotlib.pyplot as plt
# import japanize_matplotlib
import matplotlib_fontja
from utils.corr import calc_delay
from utils.es.load import load_q_and_dt_for_period
import numpy as np
from utils.init_matplotlib import figsize_px_to_inch, init_rcParams
from utils.q import Q, calc_q_kw
from utils.correlogram import (
    unify_deltas_between_dts_v2,
)
from utils.colors import colorlist


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


FONT_SIZE = 14

# > python3 calc_corr.py -dt 2022/06/02 -surface_tilt 22 -surface_azimuth 179 -show_graph
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", nargs='+', type=str)  # グラフ描画したい日付のリスト
    parser.add_argument("-surface_tilt", type=int, default=22)
    parser.add_argument("-surface_azimuth", type=float, default=185.0)
    parser.add_argument("-show_graph", action="store_true")
    args = parser.parse_args()

    for dt_str in args.dt:
        print(dt_str)
        year, month, day = dt_str.split("/")
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
            surface_tilt=args.surface_tilt,
            surface_azimuth=args.surface_azimuth,
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

        figsize_inch = figsize_px_to_inch(np.array([1280, 720]))
        plt.rcParams = init_rcParams(plt.rcParams, FONT_SIZE, figsize_inch)

        axes = [plt.subplots()[1] for _ in range(2)]

        # 実測値と計算値
        axes[0].plot(
            unified_dates,
            q_all,
            label=f"実測値",
            color=colorlist[0],
        )
        axes[0].plot(
            unified_dates,
            calced_q_all,
            label=f"計算値",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[0].set_xlabel("時刻")
        axes[0].set_ylabel("日射量 [kW/m$^2$]")
        axes[0].xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
        axes[0].legend(fontsize=20)

        # 自作の計算式で計算データを求める
        calced_q_by_original = np.vectorize(lambda dt: calc_q_kw(dt, lat, lng))(dt_all)

        # q_all = min_max(q_all)
        # calced_q_by_original = min_max(calced_q_by_original)

        (
            corr_with_real_and_calc_by_original,
            estimated_delay_with_real_and_calc_by_original,
        ) = calc_delay(calced_q_by_original, q_all)

        axes[1].plot(
            unified_dates,
            q_all,
            label=f"実測値",
            color=colorlist[0],
        )
        axes[1].plot(
            unified_dates,
            calced_q_by_original,
            label=f"計算値",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[1].set_xlabel("時刻")
        axes[1].set_ylabel("日射量 [kW/m$^2$]")
        axes[1].xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
        axes[1].legend(fontsize=20)

        print(f"ずれ時間（実測値と計算値（大気外日射量））: {estimated_delay_with_real_and_calc_by_original}[s]")
        print(f"ずれ時間（実測値と計算値（pvlib））: {estimated_delay_with_real_and_calc}[s]")

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

        # print(f"dt_all[noon_idx]: {dt_all[noon_idx]}")

        threshold_q = 0.2

        # 2.a 午前で実測値が指定した値に最も近いときのtimestampを取得する
        q_all_copy = np.copy(q_all)
        left_timestamp_idx = np.argmin(np.abs(q_all_copy[:noon_idx] - threshold_q))
        threshold_q_mask_from = dt_all[left_timestamp_idx]
        # 2.b 午後で実測値が指定した値に最も近いときのtimestampを取得する
        q_all_copy = np.copy(q_all)
        right_timestamp_idx = (
            np.argmin(np.abs(q_all_copy[noon_idx:] - threshold_q)) + noon_idx
        )
        threshold_q_mask_to = dt_all[right_timestamp_idx]

        # 3. 実測値を取得したタイムスタンプでマスキングする
        # print(f"{threshold_q_mask_from} 〜 {threshold_q_mask_to}")

        mask = (threshold_q_mask_from <= dt_all) & (dt_all < threshold_q_mask_to)

        # 実測値のthreshold_q_mask_from ~ threshold_q_mask_to以外を0に置き換える
        inverted_mask = np.logical_not(mask)
        np.putmask(q_all, inverted_mask, q_all * 0)

        (
            corr_with_real_and_calc,
            estimated_delay_with_real_and_calc,
        ) = calc_delay(calced_q_all, q_all)
        print(f"ずれ時間（実測値（前処理済み）と計算値（pvlib））: {estimated_delay_with_real_and_calc}[s]")

    if args.show_graph:
        plt.show()
