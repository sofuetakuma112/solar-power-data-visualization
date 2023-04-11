from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b1, a1 = butter_lowpass(highcut, fs, order=order)
    b2, a2 = butter_highpass(lowcut, fs, order=order)
    b = np.convolve(b1, b2)
    a = np.convolve(a1, a2)
    y = filtfilt(b, a, data)
    return y

def smooth(data, cutoff, fs):
    b, a = butter_lowpass(cutoff, fs)
    y = filtfilt(b, a, data)
    return y


# サンプルデータ
fs = 1000  # サンプリング周波数 [Hz]
t = np.arange(0, 1, 1 / fs)  # 時間軸
f1, f2 = 10, 100  # 周波数 [Hz]
data = np.sin(2 * np.pi * f1 * t) + 0.1 * np.sin(2 * np.pi * f2 * t)

# フィルタリング
lowcut = 1e-6  # ハイパスフィルターのカットオフ周波数（フィルターが通す最低周波数） [Hz]
highcut = 30  # ローパスフィルターのカットオフ周波数（フィルターが通す最高周波数） [Hz]
data_filtered_by_bandpass = bandpass_filter(data, lowcut, highcut, fs)

# フィルタリング
cutoff = 30  # カットオフ周波数 [Hz]
data_filtered = smooth(data, cutoff, fs)

# グラフ描画
fig, ax = plt.subplots(3, 1, figsize=(8, 6))
ax[0].plot(t, data)
ax[0].set_xlabel("Time [sec]")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("Raw signal")
ax[1].plot(t, data_filtered_by_bandpass)
ax[1].set_xlabel("Time [sec]")
ax[1].set_ylabel("Amplitude")
ax[1].set_title(f"Band-pass filtered ({lowcut} - {highcut} Hz)")
ax[2].plot(t, data_filtered)
ax[2].set_xlabel("Time [sec]")
ax[2].set_ylabel("Amplitude")
ax[2].set_title(f"Low-pass filtered (cutoff={cutoff} Hz)")

plt.tight_layout()
plt.show()
