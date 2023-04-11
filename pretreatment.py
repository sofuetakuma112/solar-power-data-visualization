import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils.correlogram import unify_deltas_between_dts_v2

from utils.es.load import load_q_and_dt_for_period

import matplotlib.pyplot as plt

from utils.q import Q


def bandpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2  # ナイキスト周波数
    wp = fp / fn  # ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band")  # フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
    return y  # フィルタ後の信号を返す


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

    q = Q()
    surface_tilt = 28
    surface_azimuth = 178.28
    calced_q_all = q.calc_qs_kw_v2(
        dt_all,
        latitude=33.82794,
        longitude=132.75093,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        model="isotropic",
    )

    # 1. データの正規化と窓関数の適用
    def normalize(data):
        return (data - np.mean(data)) / np.std(data)

    processing_measured_data = normalize(q_all)
    processing_theoretical_data = normalize(calced_q_all)

    plt.plot(dt_all, processing_measured_data, label="実測値")
    plt.plot(dt_all, processing_theoretical_data, label="理論値")
    plt.title("正規化の適用")
    plt.xlabel("時刻")
    plt.ylabel("日射量 [kW/m$^2$]")
    plt.legend()
    plt.show()

    window = signal.windows.hann(len(dt_all))
    processing_measured_data = processing_measured_data * window
    processing_theoretical_data = processing_theoretical_data * window

    plt.plot(dt_all, processing_measured_data, label="実測値")
    plt.plot(dt_all, processing_theoretical_data, label="理論値")
    plt.title("窓関数の適用")
    plt.xlabel("時刻")
    plt.ylabel("日射量 [kW/m$^2$]")
    plt.legend()
    plt.show()

    # 2. フィルタリング
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype="band")
        filtered_data = signal.lfilter(b, a, data)
        return filtered_data

    sample_rate = 1

    # processing_measured_data = bandpass_filter(processing_measured_data, 1e-6, 0.49, sample_rate)
    # processing_theoretical_data = bandpass_filter(
    #     processing_theoretical_data, 1e-6, 0.49, sample_rate
    # )

    # plt.plot(dt_all, processing_measured_data, label="実測値")
    # plt.plot(dt_all, processing_theoretical_data, label="理論値")
    # plt.title("バンドパスフィルターによるフィルタリング")
    # plt.xlabel("時刻")
    # plt.ylabel("日射量 [kW/m$^2$]")
    # plt.legend()
    # plt.show()

    # 3. データのトレンド除去と正規化
    processing_measured_data = signal.detrend(processing_measured_data)
    processing_theoretical_data = signal.detrend(processing_theoretical_data)

    processing_measured_data = normalize(processing_measured_data)
    processing_theoretical_data = normalize(processing_theoretical_data)

    plt.plot(dt_all, processing_measured_data, label="実測値")
    plt.plot(dt_all, processing_theoretical_data, label="理論値")
    plt.title("データのトレンド除去と正規化")
    plt.xlabel("時刻")
    plt.ylabel("日射量 [kW/m$^2$]")
    plt.legend()
    plt.show()

    # # 4. スペクトル解析とフィルタリング
    # f_measured, Pxx_den_measured = signal.periodogram(windowed_measured_data, sample_rate)
    # f_theoretical, Pxx_den_theoretical = signal.periodogram(
    #     windowed_theoretical_data, sample_rate
    # )
    # # スペクトル解析の結果をプロット
    # plt.semilogy(f_measured, Pxx_den_measured)
    # plt.semilogy(f_theoretical, Pxx_den_theoretical)
    # plt.title("スペクトル解析")
    # plt.xlabel("周波数 [Hz]")
    # plt.ylabel("パワースペクトル密度")
    # plt.show()

    # 5. 相互相関の計算
    cross_correlation = np.correlate(
        processing_measured_data,
        processing_theoretical_data,
        mode="full",
    )

    # 相互相関のピークを見つける
    peak_index = np.argmax(cross_correlation)

    # タイムスタンプのずれ時間を計算
    time_shift = peak_index - (len(q_all) - 1)

    print(f"Time shift: {time_shift}")

    # 結果をプロット
    plt.figure()
    lags = np.arange(-(len(q_all) - 1), len(q_all))  # 修正されたx軸のラグ値
    plt.plot(lags, cross_correlation)
    plt.axvline(x=0, color="r")  # x = 0 の点で縦方向に直線を引く
    plt.title("実測データと理論データの相互相関")
    plt.xlabel("ずれ時間")
    plt.ylabel("相互相関")
    plt.show()
