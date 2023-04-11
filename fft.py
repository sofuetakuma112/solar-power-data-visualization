import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils.correlogram import unify_deltas_between_dts_v2

from utils.es.load import load_q_and_dt_for_period

import matplotlib.pyplot as plt

from utils.q import Q

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

    # データを正規化する
    q_all = (q_all - np.mean(q_all)) / np.std(q_all)

    # データのFFTを計算する
    n = len(q_all)  # データの長さ
    dt = 1  # サンプリング間隔（時間単位）
    f = np.fft.fftfreq(n, dt)  # 周波数軸を計算する
    f = np.fft.fftshift(f)  # 周波数軸をシフトする
    y = np.fft.fft(q_all)  # FFTを計算する
    y = np.fft.fftshift(y)  # FFT結果をシフトする

    # パワースペクトル密度を計算する
    psd = np.abs(y) ** 2 / n * dt

    # 立ち上がりの周波数を求める
    max_psd_index = np.argmax(psd)
    max_psd_freq = f[max_psd_index]
    print("立ち上がりの箇所の周波数: ", max_psd_freq)

    # スペクトル解析結果をプロットする
    plt.plot(f, psd)
    plt.xlabel("周波数[Hz]")
    plt.ylabel("PSD")
    plt.title("日射量の1日のスペクトル解析")
    plt.xlim(-0.001, 0.001)  # x軸の範囲を設定する
    plt.show()
