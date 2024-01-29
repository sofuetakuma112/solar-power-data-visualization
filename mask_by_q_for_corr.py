import datetime
import json
import os
import matplotlib.pyplot as plt
# import japanize_matplotlib
import matplotlib_fontja
from utils.corr import calc_delay
from utils.es.load import load_q_and_dt_for_period
import argparse
import numpy as np
from utils.q import Q
from utils.correlogram import unify_deltas_between_dts_v2
from utils.colors import colorlist
import matplotlib.dates as mdates
from utils.init_matplotlib import init_rcParams, figsize_px_to_inch
import multiprocessing

# > python3 mask_by_q_for_corr.py -dt 2022/04/08 -surface_tilt 28 -surface_azimuth 178.28 -threshold_q 0.2 -corr_split_t 12:00:00 -show_fig
# > python3 mask_by_q_for_corr.py -dt 2022/06/02 -surface_tilt 22 -surface_azimuth 179.0 -threshold_q 0.2 -show_fig

FONT_SIZE = 14


def process_datetime(dt_str, split_t_str):
    year, month, day = dt_str.split("/")
    hour, minute, second = split_t_str.split(":")

    from_dt = datetime.datetime(
        int(year),
        int(month),
        int(day),
    )
    if split_t_str == "24:00:00":
        corr_split_dt = datetime.datetime(
            int(year), int(month), int(day)
        ) + datetime.timedelta(days=1)
    else:
        corr_split_dt = datetime.datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
        )

    return from_dt, corr_split_dt


def split_qs_by_idx(qs, qs_idx):
    masked_q_all_copy = np.copy(qs)
    masked_q_all_copy[:qs_idx] = 0  # 左側を0に置換
    right_masked_q_all = masked_q_all_copy

    masked_q_all_copy = np.copy(qs)
    masked_q_all_copy[qs_idx:] = 0  # 右側を0に置換
    left_masked_q_all = masked_q_all_copy

    return left_masked_q_all, right_masked_q_all


def calc_by_dt(from_dt, corr_split_dt, fig_dir_path=""):
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
    mask_from = dt_all[left_timestamp_idx]
    # 2.b 午後で実測値が指定した値に最も近いときのtimestampを取得する
    q_all_copy = np.copy(q_all)
    right_timestamp_idx = (
        np.argmin(np.abs(q_all_copy[noon_idx:] - args.threshold_q)) + noon_idx
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
    print(f"ずれ時間（指定したqだけ実測値を引いた実測データを使用）: {estimated_delay_with_subed}[s]")

    (
        _,
        estimated_delay_with_replaced_zero,
    ) = calc_delay(calced_q_all, masked_q_all_and_replaced_zero)
    print(f"ずれ時間（指定したq以下の点をすべて0に置換した実測データを使用）: {estimated_delay_with_replaced_zero}[s]")

    span = f"{mask_from.strftime('%Y-%m-%d %H:%M:%S')}〜{mask_to.strftime('%Y-%m-%d %H:%M:%S')}"

    figsize_inch = figsize_px_to_inch(np.array([1280, 720]))
    plt.rcParams = init_rcParams(plt.rcParams, FONT_SIZE, figsize_inch)

    def plot_fig(unified_dates, real_qs, calced_qs, dt_all, colorlist, title):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            unified_dates,
            real_qs,
            label=f"実測値: {dt_all[0].strftime('%Y-%m-%d')}",
            color=colorlist[0],
        )
        ax.plot(
            unified_dates,
            calced_qs,
            label=f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            linestyle="dashed",
            color=colorlist[1],
        )
        # ax.set_title(
        #     title,
        # )
        ax.set_xlabel("時刻")
        ax.set_ylabel("日射量 [kW/m$^2$]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend()
        return fig

    fig1 = plot_fig(
        unified_dates,
        masked_q_all_and_subed,
        calced_q_all,
        dt_all,
        colorlist,
        f"ずれ時間（指定したqだけ減算した実測データを使用）: {estimated_delay_with_subed}[s]\n{span}\nq: {args.threshold_q}",
    )
    if fig_dir_path != "":
        fig1.savefig(f"{fig_dir_path}/1.png")

    tmp = masked_q_all_and_subed + args.threshold_q
    inverted_mask = np.logical_not(mask)
    np.putmask(tmp, inverted_mask, tmp * np.nan)
    masked_q_all_and_added = tmp

    fig2 = plot_fig(
        unified_dates,
        masked_q_all_and_added,
        calced_q_all,
        dt_all,
        colorlist,
        f"実測値をしきい値のqだけ上にスライドさせたもの\n{span}以外は非表示にしている",
    )
    if fig_dir_path != "":
        fig2.savefig(f"{fig_dir_path}/2.png")

    fig3 = plot_fig(
        unified_dates,
        masked_q_all_and_replaced_zero,
        calced_q_all,
        dt_all,
        colorlist,
        f"ずれ時間（指定したq以下の点をすべて0に置換した実測データを使用）: {estimated_delay_with_replaced_zero}[s]\n{span}\nq: {args.threshold_q}",
    )
    if fig_dir_path != "":
        fig3.savefig(f"{fig_dir_path}/3.png")

    fig4 = plot_fig(
        unified_dates,
        q_all_raw,
        calced_q_all,
        dt_all,
        colorlist,
        f"実測データと計算データ比較用",
    )
    if fig_dir_path != "":
        fig4.savefig(f"{fig_dir_path}/4.png")

    # 1日すべて使う + 理論データも指定したしきい値だけ減算
    calced_q_all_subbed = calced_q_all - args.threshold_q
    calced_q_all_subbed[calced_q_all_subbed < 0] = 0
    (
        _,
        ed_with_subed_real_and_calc,
    ) = calc_delay(calced_q_all_subbed, masked_q_all_and_subed)
    print(f"ずれ時間（指定したqだけ減算した実測データと理論データを使用）: {ed_with_subed_real_and_calc}[s]")

    fig5 = plot_fig(
        unified_dates,
        masked_q_all_and_subed,
        calced_q_all_subbed,
        dt_all,
        colorlist,
        f"ずれ時間（指定したqだけ減算した実測データと理論データを使用）: {ed_with_subed_real_and_calc}[s]\n{span}\nq: {args.threshold_q}",
    )
    if fig_dir_path != "":
        fig5.savefig(f"{fig_dir_path}/5.png")

    def plot_qs_and_calced_qs_and_slided_qs(
        ax,
        dts,
        qs,
        calced_qs,
        label_qs,
        label_calced,
        title,
        colorlist,
        estimated_delay,
    ):
        ax.plot(dts, qs, label=label_qs, color=colorlist[0])
        ax.plot(
            dts,
            calced_qs,
            label=label_calced,
            linestyle="dashed",
            color=colorlist[1],
        )
        ax.plot(
            np.array(dts, dtype="datetime64[s]")
            + estimated_delay.astype("timedelta64[s]"),
            qs,
            label="ずれ時間だけ実測データをスライド",
            linestyle="dashed",
            color=colorlist[2],
        )
        ax.set_title(title)
        ax.set_xlabel("時刻")
        ax.set_ylabel("日射量 [kW/m$^2$]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend()

    def plot_corr(ax, lagtimes, corr, estimated_delay, colorlist):
        ax.plot(lagtimes, corr, label="相互相関", color=colorlist[0])
        ax.axvline(
            x=estimated_delay, linestyle="--", color=colorlist[1], label="推定ずれ時間"
        )
        ax.set_title(f"ずれ時間: {estimated_delay}[s]\n{span}\nq: {args.threshold_q}")
        ax.set_xlabel("ラグ時間")
        ax.set_ylabel("相互相関")
        ax.legend()

    split_indices = np.where(dt_all == corr_split_dt)[0]  # corr_split_dtのidx
    if len(split_indices) != 0:
        split_idx = split_indices[0]

        left_masked_q_all, right_masked_q_all = split_qs_by_idx(masked_q_all, split_idx)

        right_masked_q_all_subbed = right_masked_q_all - args.threshold_q
        left_masked_q_all_subbed = left_masked_q_all - args.threshold_q

        right_masked_q_all_subbed[right_masked_q_all_subbed < 0] = 0
        left_masked_q_all_subbed[left_masked_q_all_subbed < 0] = 0

        corr_left, ed_left_subed_real = calc_delay(
            calced_q_all, left_masked_q_all_subbed
        )
        corr_right, ed_right_subed_real = calc_delay(
            calced_q_all, right_masked_q_all_subbed
        )
        print(f"ずれ時間（指定したqだけ減算した実測データの左側を使用）: {ed_left_subed_real}[s]")
        print(f"ずれ時間（指定したqだけ減算した実測データの右側を使用）: {ed_right_subed_real}[s]")

        # 指定した時間で区切って2つ相互相関を求める + 実測データをしきい値分減算する
        fig6, (ax6_1, ax6_2) = plt.subplots(1, 2)

        plot_qs_and_calced_qs_and_slided_qs(
            ax6_1,
            unified_dates,
            left_masked_q_all_subbed,
            calced_q_all,
            f"実測値(corr_split_dtの左側): {dt_all[0].strftime('%Y-%m-%d')}",
            f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            f"実測データをしきい値分減算\nずれ時間: {ed_left_subed_real}[s]\n{span}\nq: {args.threshold_q}",
            colorlist,
            ed_left_subed_real,
        )
        plot_qs_and_calced_qs_and_slided_qs(
            ax6_2,
            unified_dates,
            right_masked_q_all_subbed,
            calced_q_all,
            f"実測値(corr_split_dtの右側): {dt_all[0].strftime('%Y-%m-%d')}",
            f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            f"実測データをしきい値分減算\nずれ時間: {ed_right_subed_real}[s]\n{span}\nq: {args.threshold_q}",
            colorlist,
            ed_right_subed_real,
        )
        if fig_dir_path != "":
            fig6.savefig(f"{fig_dir_path}/6.png")

        fig6_corr, (ax6_corr1, ax6_corr2) = plt.subplots(1, 2)

        x_left = np.arange(-len(left_masked_q_all_subbed) + 1, len(calced_q_all))
        plot_corr(ax6_corr1, x_left, corr_left, ed_left_subed_real, colorlist)

        x_right = np.arange(-len(calced_q_all) + 1, len(right_masked_q_all_subbed))
        plot_corr(ax6_corr2, x_right, corr_right, ed_right_subed_real, colorlist)
        if fig_dir_path != "":
            fig6_corr.savefig(f"{fig_dir_path}/6_corr.png")

        # 指定した時間で区切って2つ相互相関を求める + 実測データと計算データをしきい値分減算する
        corr_left, ed_left_subed_real_and_calc = calc_delay(
            calced_q_all_subbed, left_masked_q_all_subbed
        )
        corr_right, ed_right_subed_real_and_calc = calc_delay(
            calced_q_all_subbed, right_masked_q_all_subbed
        )
        print(
            f"ずれ時間（指定したqだけ実測データと理論データを減算した上で、実測データの左側を使用）: {ed_left_subed_real_and_calc}[s]"
        )
        print(
            f"ずれ時間（指定したqだけ実測データと理論データを減算した上で、実測データの右側を使用）: {ed_right_subed_real_and_calc}[s]"
        )

        fig7, (ax7_1, ax7_2) = plt.subplots(1, 2)
        plot_qs_and_calced_qs_and_slided_qs(
            ax7_1,
            unified_dates,
            left_masked_q_all_subbed,
            calced_q_all_subbed,
            f"実測値(corr_split_dtの左側): {dt_all[0].strftime('%Y-%m-%d')}",
            f"計算値(しきい値だけ減算): {dt_all[0].strftime('%Y-%m-%d')}",
            f"実測データと計算データをしきい値分減算\nずれ時間: {ed_left_subed_real_and_calc}[s]\n{span}\nq: {args.threshold_q}",
            colorlist,
            ed_left_subed_real_and_calc,
        )
        plot_qs_and_calced_qs_and_slided_qs(
            ax7_2,
            unified_dates,
            right_masked_q_all_subbed,
            calced_q_all_subbed,
            f"実測値(corr_split_dtの右側): {dt_all[0].strftime('%Y-%m-%d')}",
            f"計算値(しきい値だけ減算): {dt_all[0].strftime('%Y-%m-%d')}",
            f"実測データと計算データをしきい値分減算\nずれ時間: {ed_right_subed_real_and_calc}[s]\n{span}\nq: {args.threshold_q}",
            colorlist,
            ed_right_subed_real_and_calc,
        )
        if fig_dir_path != "":
            fig7.savefig(f"{fig_dir_path}/7.png")

        fig7_corr, (ax7_corr_left, ax7_corr_right) = plt.subplots(1, 2)

        x_left = np.arange(-len(left_masked_q_all_subbed) + 1, len(calced_q_all_subbed))
        plot_corr(
            ax7_corr_left, x_left, corr_left, ed_left_subed_real_and_calc, colorlist
        )

        x_right = np.arange(
            -len(calced_q_all_subbed) + 1, len(right_masked_q_all_subbed)
        )
        plot_corr(
            ax7_corr_right, x_right, corr_right, ed_right_subed_real_and_calc, colorlist
        )
        if fig_dir_path != "":
            fig7_corr.savefig(f"{fig_dir_path}/7_corr.png")

        left_masked_q_all, right_masked_q_all = split_qs_by_idx(
            masked_q_all_and_replaced_zero, split_idx
        )

        # 指定した時間で区切って2つ相互相関を求める
        corr_left, ed_left = calc_delay(calced_q_all, left_masked_q_all)
        corr_right, ed_right = calc_delay(calced_q_all, right_masked_q_all)
        print(f"ずれ時間（指定したq以下の点をすべて0に置換した実測データの左側を使用）: {ed_left}[s]")
        print(f"ずれ時間（指定したq以下の点をすべて0に置換した実測データの右側を使用）: {ed_right}[s]")

        fig8, (ax8_1, ax8_2) = plt.subplots(1, 2)
        plot_qs_and_calced_qs_and_slided_qs(
            ax8_1,
            unified_dates,
            left_masked_q_all,
            calced_q_all,
            f"実測値(corr_split_dtの左側): {dt_all[0].strftime('%Y-%m-%d')}",
            f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            f"しきい値以下の実測データを0に置換\nずれ時間: {ed_left}[s]\n{span}\nq: {args.threshold_q}",
            colorlist,
            ed_left,
        )
        plot_qs_and_calced_qs_and_slided_qs(
            ax8_2,
            unified_dates,
            right_masked_q_all,
            calced_q_all,
            f"実測値(corr_split_dtの右側): {dt_all[0].strftime('%Y-%m-%d')}",
            f"計算値: {dt_all[0].strftime('%Y-%m-%d')}",
            f"しきい値以下の実測データを0に置換\nずれ時間: {ed_right}[s]\n{span}\nq: {args.threshold_q}",
            colorlist,
            ed_right,
        )
        if fig_dir_path != "":
            fig8.savefig(f"{fig_dir_path}/8.png")

        fig8_corr, (ax8_corr_left, ax8_corr_right) = plt.subplots(1, 2)

        x_left = np.arange(-len(left_masked_q_all) + 1, len(calced_q_all))
        plot_corr(ax8_corr_left, x_left, corr_left, ed_left, colorlist)

        x_right = np.arange(-len(calced_q_all) + 1, len(right_masked_q_all))
        plot_corr(ax8_corr_right, x_right, corr_right, ed_right, colorlist)
        if fig_dir_path != "":
            fig8_corr.savefig(f"{fig_dir_path}/8_corr.png")

    plt.tight_layout()

    if args.show_fig:
        plt.show()


def create_dir_path(split_t_str, th_q, dt_str):
    # splitに使用したtime: st
    # 理論データに対して減算した値: tsv
    # 実測データに対して減算した値: rsv
    # しきい値のq: thq
    fig_dir_path = f"{OUTPUT_DIR_PATH}/st-{split_t_str}_thq-{th_q}/{dt_str.replace('/', '-')}"  # 図の保存先ディレクトリ

    return fig_dir_path


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
    parser.add_argument(
        "-corr_split_t", type=str, default="24:00:00"
    )  # 相互相関の計算を分割するタイムスタンプ
    parser.add_argument("-sub_calc_q", action="store_true")  # 計算値もしきい値分減算するか
    parser.add_argument("-show_fig", action="store_true")
    parser.add_argument("-save_fig", action="store_true")
    args = parser.parse_args()

    if args.dt == None:
        # json_open = open(USER_INPUT_JSON_FILE_PATH, "r")
        # mask_from_tos = json.load(json_open)

        # for i, from_dt_str in enumerate(mask_from_tos.keys()):
        #     from_dt, corr_split_dt = process_datetime(from_dt_str, args.corr_split_t)

        #     fig_dir_path = create_dir_path(
        #         args.corr_split_t, args.threshold_q, from_dt_str
        #     )

        #     if os.path.exists(fig_dir_path):
        #         continue
        #     else:
        #         os.makedirs(fig_dir_path)
        #         calc_by_dt(from_dt, corr_split_dt, fig_dir_path)

        # jsonから読み込む
        json_open = open(USER_INPUT_JSON_FILE_PATH, "r")
        mask_from_tos = json.load(json_open)

        def process_mask(from_dt_str):
            from_dt, corr_split_dt = process_datetime(from_dt_str, args.corr_split_t)

            fig_dir_path = create_dir_path(
                args.corr_split_t, args.threshold_q, from_dt_str
            )

            if os.path.exists(fig_dir_path):
                return
            else:
                os.makedirs(fig_dir_path)
                calc_by_dt(from_dt, corr_split_dt, fig_dir_path)

        with multiprocessing.Pool() as pool:
            pool.map(process_mask, mask_from_tos.keys())
    else:
        from_dt, corr_split_dt = process_datetime(args.dt, args.corr_split_t)

        if args.save_fig:
            fig_dir_path = create_dir_path(args.corr_split_t, args.threshold_q, args.dt)
            if not os.path.exists(fig_dir_path):
                os.makedirs(fig_dir_path)
        else:
            fig_dir_path = ""

        calc_by_dt(from_dt, corr_split_dt, fig_dir_path)
