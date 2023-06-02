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
        yf: ndarray - 振幅（複素数）
    """
    N = len(data) # 時系列データの長さ
    T = 1 / sampling_rate # サンプリング周期
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
    入力された yf_measured と yf_calculated の中で最も強い周波数成分を見つけ、その周波数成分における位相を計算しています。そして、それぞれの位相差を計算して返しています。

    入力:
        yf_measured: 実測値のデータ
        yf_calculated: 計算値のデータ
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