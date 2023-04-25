import datetime

from matplotlib import pyplot as plt
import japanize_matplotlib
from utils.es.load import load_q_and_dt_for_period
import argparse
from utils.q import Q
from utils.correlogram import unify_deltas_between_dts_v2
import numpy as np
from scipy.fft import fft, fftfreq


def perform_fft(data, sampling_rate):
    """
    概要:
        与えられた時系列データをFFT変換し、その周波数成分の振幅と周波数を返す

    引数:
        data: list, ndarray - 時系列データ
        sampling_rate: float - サンプリング周波数

    返り値:
        xf: ndarray - 周波数軸
        np.abs(yf[0 : N // 2]): ndarray - 周波数成分の振幅
    """
    N = len(data)
    T = 1 / sampling_rate
    yf = fft(data)
    xf = fftfreq(N, T)[: N // 2]
    return xf, yf


def find_fundamental_frequency(xf, yf, N):
    """
    概要:
        フーリエ変換により得られた周波数スペクトルから、最大振幅を示す周波数を見つける関数

    引数:
        xf(numpy.ndarray): 離散フーリエ変換によって得られた振幅スペクトルの周波数軸上の値が格納された配列
        yf(numpy.ndarray): 離散フーリエ変換によって得られた振幅スペクトルの振幅が格納された配列

    戻り値:
        float: 振幅スペクトルの振幅が最大となる周波数
    """
    max_index = np.argmax(np.abs(yf[0 : N // 2]))  # 最大振幅を持つ周波数スペクトルデータのインデックスを取得する
    return xf[max_index]


def find_phase_difference(yf_measured, yf_calculated, N):
    """
    2つの時系列データから位相差を計算する関数
    入力された measured_data と calculated_data の中で最も強い周波数成分を見つけ、その周波数成分における位相を計算しています。そして、それぞれの位相差を計算して返しています。

    入力:
        measured_data: 実測値のデータ
        calculated_data: 計算値のデータ
        N: 時系列データの長さ

    出力:
        phase_diff: 実測値と計算値の位相差
    """
    # np.argmax(yf_measured)で振幅が最大を取る点のインデックスを求める
    # yf_measured[np.argmax(yf_measured)]で振幅の最大値を求める
    # np.angleで振幅の最大値（基本周波数）に対応する位相を求める
    phase_measured = np.angle(yf_measured[np.argmax(np.abs(yf_measured[0 : N // 2]))])
    phase_calculated = np.angle(
        yf_calculated[np.argmax(np.abs(yf_calculated[0 : N // 2]))]
    )
    print(f"実測データの基本周波数に対応する位相: {phase_measured} [rad]")
    print(f"計算データの基本周波数に対応する位相: {phase_calculated} [rad]")

    phase_diff = phase_measured - phase_calculated
    print(f"位相差: {phase_diff} [rad]")

    return phase_diff


# 位相から時間へ変換する
def phase_to_time(phase_diff, fundamental_freq):
    return phase_diff / (2 * np.pi * fundamental_freq)


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

    # 実測データと計算データにFFTを実行する
    q_all = q_all - np.mean(q_all)
    xf_measured, yf_measured = perform_fft(q_all, sampling_rate)
    fundamental_freq_measured = find_fundamental_frequency(
        xf_measured, yf_measured, len(q_all)
    )
    print(f"実測データの基本周波数: {fundamental_freq_measured} [Hz]")

    phase_diffs = np.array([])
    shifts = np.arange(0 + args.shift_offset, args.max_shift, 1)
    for shift in shifts:
        calced_q_all_rolled = np.roll(calced_q_all, shift)
        calced_q_all_rolled = calced_q_all_rolled - np.mean(calced_q_all_rolled)
        # 計算データをFFTする
        xf_calculated, yf_calculated = perform_fft(calced_q_all_rolled, sampling_rate)
        # 基本周波数を求める
        fundamental_freq_calculated = find_fundamental_frequency(
            xf_calculated, yf_calculated, len(calced_q_all_rolled)
        )
        print(f"計算データの基本周波数: {fundamental_freq_calculated} [Hz]")

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

    fig, ax = plt.subplots()
    ax.plot(shifts, phase_diffs, label="ずれ時間とそれに対応する位相差")
    ax.set_xlabel("ずれ時間 [s]")
    ax.set_ylabel("位相差 [rad]")
    ax.legend()
    plt.show()
