import datetime
import json
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.corr import calc_delay
from utils.es.load import load_q_and_dt_for_period
import argparse
import numpy as np
from utils.q import Q
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from utils.init_matplotlib import init_rcParams, figsize_px_to_inch

# > python3 mask_by_q_for_corr.py -dt 2022/04/08 -surface_tilt 28 -surface_azimuth 178.28 -threshold_q 0.2 -bundle

FONT_SIZE = 14


def calc_by_dt(from_dt, fig_image_path=""):
    diff_days = 1.0
    dt_all, q_all = load_q_and_dt_for_period(from_dt, diff_days)
    dt_all, q_all = unify_deltas_between_dts_v2(dt_all, q_all)

    q_all_raw = np.copy(q_all)

    # 昇順で並んでいるかテストする
    sort_indexes = np.argsort(dt_all)
    if not np.allclose(sort_indexes, np.arange(0, dt_all.size, 1)):
        raise ValueError("dt_allが時系列順で並んでいない")

    q = Q()  # インスタンス作成時にDBへのコネクションを初期化
    calced_q_all = q.calc_qs_kw_v2(
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model=args.model,
    )

    # 1. 12時の左側と右側でそれぞれ1点ずつ指定したqの値に最も近い点のタイムスタンプを探す
    diffs_from_noon = dt_all - datetime.datetime(
        int(year), int(month), int(day), 12, 0, 0
    )
    noon_idx = np.argmin(
        np.vectorize(lambda diff_delta: np.abs(diff_delta.total_seconds()))(
            diffs_from_noon
        )
    )

    print(f"dt_all[noon_idx]: {dt_all[noon_idx]}")

    # 2.a 午前で実測値が指定した値に最も近いときのtimestampを取得する
    left_timestamp_idx = np.argmin(np.abs(q_all[:noon_idx] - args.threshold_q))
    mask_from = dt_all[left_timestamp_idx]
    # 2.b 午後で実測値が指定した値に最も近いときのtimestampを取得する
    right_timestamp_idx = (
        np.argmin(np.abs(q_all[noon_idx:] - args.threshold_q)) + noon_idx
    )
    mask_to = dt_all[right_timestamp_idx]

    print(f"mask_from: {mask_from}")
    print(f"mask_to: {mask_to}")

    # 3. 実測値を取得したタイムスタンプでマスキングする
    print(f"{mask_from} 〜 {mask_to}")

    mask = (mask_from <= dt_all) & (dt_all < mask_to)

    # 実測値のmask_from ~ mask_to以外を0に置き換える
    inverted_mask = np.logical_not(mask)
    np.putmask(q_all, inverted_mask, q_all * 0)

    masked_q_all = q_all
    masked_dt_all = dt_all

    # mask_from, mask_toの外側をすべての0にした
    masked_q_all_and_replaced_zero = np.copy(masked_q_all)

    # mask_from ~ mask_toのすべての点で「実測値列 - 指定したq」を求めて、マイナスになった箇所は0にする
    masked_q_all_and_subed = masked_q_all - args.threshold_q
    masked_q_all_and_subed[masked_q_all_and_subed < 0] = 0

    unified_dates = np.vectorize(
        lambda dt: datetime.datetime(
            2022, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond
        )
    )(masked_dt_all)

    # np.correlate(M, N): Mが0パディングされる側、Nがスライドする側
    (
        _,
        estimated_delay_with_subed,
    ) = calc_delay(calced_q_all, masked_q_all_and_subed)
    print(f"ずれ時間（指定したqだけ実測値を引いて最小値0にした実測データを使用）: {estimated_delay_with_subed}[s]")

    (
        _,
        estimated_delay_with_replaced_zero,
    ) = calc_delay(calced_q_all, masked_q_all_and_replaced_zero)
    print(f"ずれ時間（指定したq以下の点をすべて0に置換した実測データを使用）: {estimated_delay_with_replaced_zero}[s]")

    span = f"{mask_from.strftime('%Y-%m-%d %H:%M:%S')}〜{mask_to.strftime('%Y-%m-%d %H:%M:%S')}"

    if args.bundle:
        # 一枚にまとめてプロット
        # 相互相関を求める時
        figsize_inch = figsize_px_to_inch(np.array([1920, 1080]))
        plt.rcParams = init_rcParams(plt.rcParams, 16, figsize_inch)

        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(
            unified_dates,
            masked_q_all_and_subed,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[0, 0].plot(
            unified_dates,
            calced_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[0, 0].set_title(
            f"ずれ時間: {estimated_delay_with_subed}[s]\n{span}\nq: {args.threshold_q}",
        )
        axes[0, 0].set_xlabel("時刻")
        axes[0, 0].set_ylabel("日射量 [kW/m$^2$]")
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[0, 0].legend()

        tmp = masked_q_all_and_subed + args.threshold_q
        inverted_mask = np.logical_not(mask)
        np.putmask(tmp, inverted_mask, tmp * np.nan)
        masked_q_all_and_subed = tmp

        # 実測値と計算値
        axes[0, 1].plot(
            unified_dates,
            masked_q_all_and_subed,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[0, 1].plot(
            unified_dates,
            calced_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[0, 1].set_title(f"実測値をしきい値のqだけ上にスライドさせたもの\n{span}以外は非表示にしている")
        axes[0, 1].set_xlabel("時刻")
        axes[0, 1].set_ylabel("日射量 [kW/m$^2$]")
        axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[0, 1].legend()

        axes[1, 0].plot(
            unified_dates,
            masked_q_all_and_replaced_zero,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[1, 0].plot(
            unified_dates,
            calced_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[1, 0].set_title(
            f"ずれ時間: {estimated_delay_with_replaced_zero}[s]\n{span}\nq: {args.threshold_q}",
        )
        axes[1, 0].set_xlabel("時刻")
        axes[1, 0].set_ylabel("日射量 [kW/m$^2$]")
        axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[1, 0].legend()

        axes[1, 1].plot(
            unified_dates,
            q_all_raw,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        axes[1, 1].plot(
            unified_dates,
            calced_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        axes[1, 1].set_title(
            f"実測データと計算データ比較用",
        )
        axes[1, 1].set_xlabel("時刻")
        axes[1, 1].set_ylabel("日射量 [kW/m$^2$]")
        axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[1, 1].legend()

        plt.tight_layout()

        if fig_image_path == "":
            plt.show()
        else:
            # 新しい figure を画像として保存する
            plt.savefig(fig_image_path)
    else:
        # Figure 1
        figsize_inch = figsize_px_to_inch(np.array([1280, 720]))
        plt.rcParams = init_rcParams(plt.rcParams, FONT_SIZE, figsize_inch)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(
            unified_dates,
            masked_q_all_and_subed,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        ax1.plot(
            unified_dates,
            calced_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        ax1.set_title(
            f"ずれ時間: {estimated_delay_with_subed}[s]\n{span}\nq: {args.threshold_q}",
        )
        ax1.set_xlabel("時刻")
        ax1.set_ylabel("日射量 [kW/m$^2$]")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.legend()
        # fig1.savefig("figure1.png")

        # Figure 2
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        tmp = masked_q_all_and_subed + args.threshold_q
        inverted_mask = np.logical_not(mask)
        np.putmask(tmp, inverted_mask, tmp * np.nan)
        masked_q_all_and_subed = tmp

        ax2.plot(
            unified_dates,
            masked_q_all_and_subed,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        ax2.plot(
            unified_dates,
            calced_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        ax2.set_title(f"実測値と計算値の概形がどの程度一致しているか確認用")
        ax2.set_xlabel("時刻")
        ax2.set_ylabel("日射量 [kW/m$^2$]")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.legend()
        # fig2.savefig("figure2.png")

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)

        ax3.plot(
            unified_dates,
            masked_q_all_and_replaced_zero,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        ax3.plot(
            unified_dates,
            calced_q_all,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        ax3.set_title(
            f"ずれ時間: {estimated_delay_with_replaced_zero}[s]\n{span}\nq: {args.threshold_q}",
        )
        ax3.set_xlabel("時刻")
        ax3.set_ylabel("日射量 [kW/m$^2$]")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax3.legend()

        plt.show()


USER_INPUT_JSON_FILE_PATH = f"data/json/calc_corr_by_day/user_input.json"
OUTPUT_DIR_PATH = "images/mask_by_q_for_corr"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    parser.add_argument(
        "-model", type=str, default="isotropic"
    )  # 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    parser.add_argument("-surface_tilt", type=int, default=22)
    parser.add_argument("-surface_azimuth", type=float, default=185.0)
    parser.add_argument("-threshold_q", type=float, default=0.2)
    parser.add_argument("-bundle", action="store_true")
    args = parser.parse_args()

    if args.dt == None:
        # jsonから読み込む
        json_open = open(USER_INPUT_JSON_FILE_PATH, "r")
        mask_from_tos = json.load(json_open)

        fig_dir_path = f"{OUTPUT_DIR_PATH}"  # 図の保存先ディレクトリ
        if not os.path.exists(fig_dir_path):
            os.makedirs(fig_dir_path)

        for i, from_dt_str in enumerate(mask_from_tos.keys()):
            year, month, day = from_dt_str.split("/")
            from_dt = datetime.datetime(
                int(year),
                int(month),
                int(day),
            )

            fig_image_path = (
                f"{fig_dir_path}/{str(i).zfill(4)}: {from_dt.strftime('%Y-%m-%d')}.png"
            )

            if os.path.isfile(fig_image_path):
                continue
            else:
                calc_by_dt(from_dt, fig_image_path)
    else:
        year, month, day = args.dt.split("/")
        from_dt = datetime.datetime(
            int(year),
            int(month),
            int(day),
        )

        calc_by_dt(from_dt)
