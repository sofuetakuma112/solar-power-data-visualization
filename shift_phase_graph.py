import datetime

from matplotlib import pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import argparse
from utils.frequency_analysis import find_phase_difference, perform_fft
from utils.q import Q
from utils.correlogram import unify_deltas_between_dts_v2
import numpy as np
from scipy.fft import fft, fftfreq

# > python3 phase_difference/shift_phase_graph.py -dt 2022/04/08 -max_shift 1000
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付のリスト
    parser.add_argument(
        "-model", type=str, default="isotropic"
    )  # 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'
    parser.add_argument("-surface_tilt", type=int, default=22)
    parser.add_argument("-surface_azimuth", type=float, default=185.0)
    parser.add_argument("-shift_offset", type=int, default=0)
    parser.add_argument("-max_shift", type=int, default=10)
    parser.add_argument("-plot_fft_flag", action="store_true")
    parser.add_argument("-plot_q_flag", action="store_true")
    args = parser.parse_args()

    year, month, day = args.dt.split("/")
    from_dt = datetime.datetime(
        int(year),
        int(month),
        int(day),
    )

    diff_days = 1.0  # 開始の日付から何日間分の実装データを読み込むか
    dt_all, q_all = load_q_and_dt_for_period(
        from_dt, diff_days
    )  # from_dtからdiff_days日分の日射量の実測データを読み込む
    dt_all, q_all = unify_deltas_between_dts_v2(
        dt_all, q_all
    )  # 実測データのタイムスタンプ間隔が1.0sになるようにリサンプリングする

    q = Q()  # 与えられたタイムスタンプ列を引数として、日射量モデルから快晴時の日射量を求める処理を行うクラスを初期化
    calced_q_all = q.calc_qs_kw_v2(  # 日射量モデルによる日射量の計算データを求める
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        model=args.model,
    )

    # サンプリングレートを設定する
    sampling_rate = 1.0

    # 実測データをFFTする
    q_all = q_all - np.mean(q_all)
    xf_measured, yf_measured = perform_fft(q_all, sampling_rate)

    phase_diffs = np.array([])
    shifts = np.arange(0 + args.shift_offset, args.max_shift, 1)
    for shift in shifts:
        calced_q_all_rolled = np.roll(calced_q_all, shift)  # 計算データをshift分遅らせる

        # 計算データをFFTする
        calced_q_all_rolled = calced_q_all_rolled - np.mean(calced_q_all_rolled)
        xf_calculated, yf_calculated = perform_fft(calced_q_all_rolled, sampling_rate)

        # FFTの結果をプロット
        if args.plot_fft_flag:
            fig, ax = plt.subplots()
            ax.plot(xf_measured, yf_measured, label="実測データ")
            ax.plot(xf_calculated, yf_calculated, label="計算データ")
            ax.set_xlabel("周波数 [Hz]")
            ax.set_ylabel("振幅")
            ax.legend()
            plt.show()

        if args.plot_q_flag:
            fig, ax = plt.subplots()
            ax.plot(dt_all, q_all, label="実測データ")
            ax.plot(dt_all, calced_q_all_rolled, label="計算データ")
            ax.set_xlabel("日時")
            ax.set_ylabel("日射量")
            ax.legend()
            plt.show()

        # 位相差を計算する
        phase_diff = find_phase_difference(yf_measured, yf_calculated, len(yf_measured))

        phase_diffs = np.append(phase_diffs, phase_diff)

    # 横軸ずれ時間、縦軸位相差のグラフの傾きを求める
    slope, intercept = np.polyfit(shifts, phase_diffs, 1)
    print(f"1sずれるごとに増えるラジアン: {slope} [rad]")  # 7.272289386659088e-05 [rad]

    fig, ax = plt.subplots()
    ax.plot(shifts, phase_diffs, label="ずれ時間とそれに対応する位相差")
    ax.set_xlabel("ずれ時間 [s]")
    ax.set_ylabel("位相差 [rad]")
    ax.legend()
    plt.show()
