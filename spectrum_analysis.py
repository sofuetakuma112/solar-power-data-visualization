import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils.correlogram import unify_deltas_between_dts_v2

from utils.es.load import load_q_and_dt_for_period

import matplotlib.pyplot as plt


def remove_dc_component(data):
    return data - np.mean(data)


def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


def apply_window_function(data, window_type="hann"):
    if window_type == "hann":
        window = signal.windows.hann(len(data))
    elif window_type == "hamming":
        window = signal.windows.hamming(len(data))
    else:
        raise ValueError("Invalid window_type specified.")
    return data * window


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", type=str)  # グラフ描画したい日付
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

    # q_all = q_all * 1000 # kW => Wに変換

    # 前処理
    q_cleaned = remove_dc_component(q_all)
    q_normalized = normalize_data(q_cleaned)
    q_windowed = apply_window_function(q_normalized, window_type="hann")

    plt.plot(dt_all, q_windowed)
    plt.xlabel("Time (hours)")
    plt.ylabel("Windowed Solar Radiation")
    plt.title("Windowed Solar Radiation vs Time")
    plt.grid(True)
    plt.show()

    fs = 1  # サンプリング周波数（Hz）

    # スペクトル解析（パワースペクトル密度を求める）
    f, Pxx_den = signal.periodogram(q_windowed, fs)

    # プロット
    plt.semilogy(f, Pxx_den)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectral density")
    plt.title("Spectral Analysis")
    plt.grid(True)
    plt.show()
