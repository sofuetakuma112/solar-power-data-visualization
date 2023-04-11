import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# データの生成（例としてサイン波とコサイン波を使用）
t = np.linspace(0, 10, 1000)  # xs
measured_data = np.sin(2 * np.pi * 5 * t)  # ys1
theoretical_data = np.cos(2 * np.pi * 5 * t)  # ys2

# プロット
plt.plot(t, measured_data, label="Measured data (sin wave)")
plt.plot(t, theoretical_data, label="Theoretical data (cos wave)")

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Sin wave and Cos wave")
plt.legend()
plt.grid(True)
plt.show()

# 1. データの正規化と窓関数の適用
def normalize(data):
    return (data - np.mean(data)) / np.std(data)


normalized_measured_data = normalize(measured_data)
normalized_theoretical_data = normalize(theoretical_data)

window = signal.windows.hann(len(t))
windowed_measured_data = normalized_measured_data * window
windowed_theoretical_data = normalized_theoretical_data * window


# 2. フィルタリングとデータの補間
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist周波数（サンプリング周波数の半分）を計算します。これは、デジタル信号に対する最大の周波数成分です。
    low = lowcut / nyq  # 正規化された下限周波数を計算します。これは、パスバンドの開始点を表します。
    high = highcut / nyq  # 正規化された上限周波数を計算します。これは、パスバンドの終了点を表します。
    b, a = signal.butter(
        order, [low, high], btype="band"
    )  # order次数のバターワース・バンドパスフィルターの係数b（分子係数）とa（分母係数）を計算します。
    filtered_data = signal.lfilter(
        b, a, data
    )  # 計算されたフィルター係数bとaを使用して、入力データdataにバンドパスフィルターを適用し、フィルタリングされたデータfiltered_dataを得ます。signal.lfilterは、SciPyのsignalモジュールに含まれる関数で、線形フィルタリングを実行します。
    return filtered_data


filtered_measured_data = bandpass_filter(measured_data, 4, 6, 100)
filtered_theoretical_data = bandpass_filter(theoretical_data, 4, 6, 100)

# 3. データのトレンド除去と正規化
detrend_measured_data = signal.detrend(measured_data)
detrend_theoretical_data = signal.detrend(theoretical_data)

normalized_detrend_measured_data = normalize(detrend_measured_data)
normalized_detrend_theoretical_data = normalize(detrend_theoretical_data)

# 4. スペクトル解析とフィルタリング
f_measured, Pxx_den_measured = signal.periodogram(measured_data, fs=100)
f_theoretical, Pxx_den_theoretical = signal.periodogram(theoretical_data, fs=100)

# スペクトル解析の結果をプロット
plt.semilogy(f_measured, Pxx_den_measured)
plt.semilogy(f_theoretical, Pxx_den_theoretical)
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD")
plt.show()

# 周波数帯域を特定し、フィルターを適用
filtered_measured_data = bandpass_filter(measured_data, 4, 6, 100)
filtered_theoretical_data = bandpass_filter(theoretical_data, 4, 6, 100)

# 5. 相互相関の計算
cross_correlation = np.correlate(
    filtered_measured_data, filtered_theoretical_data, mode="full"
)

# 相互相関のピークを見つける
peak_index = np.argmax(cross_correlation)

# タイムスタンプのずれ時間を計算
time_shift = peak_index - (len(measured_data) - 1)

print(f"Time shift: {time_shift}")

# 結果をプロット
plt.figure()
lags = np.arange(-(len(measured_data) - 1), len(measured_data))  # 修正されたx軸のラグ値
plt.plot(lags, cross_correlation)
plt.title("Cross-correlation between measured and theoretical data")
plt.xlabel("Lag")
plt.ylabel("Cross-correlation")
plt.show()
