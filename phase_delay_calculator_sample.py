from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# 信号生成
sampling_rate = 1000
time = np.linspace(0, 1, sampling_rate) # sampling_rateのサンプリングレートでサンプリング
frequency = 5
phase_shift = 0.05  # 2つの信号間のずれ時間（指定したphase_shiftだけsignal2がsignal1に対して遅れる）
signal1 = np.sin(2 * np.pi * frequency * time)
signal2 = np.sin(2 * np.pi * frequency * (time - phase_shift))

plt.figure(figsize=(10, 5))
plt.plot(time, signal1, label="Signal 1")
plt.plot(time, signal2, label="Signal 2")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# FFT
N = len(signal1)
T = 1 / sampling_rate
yf1 = fft(signal1)
yf2 = fft(signal2)
xf = fftfreq(N, T)[: N // 2]

# 基本周波数と対応するインデックスを求める
fundamental_freq_index = np.argmax(np.abs(yf1[0 : N // 2]))

# 位相差を求める
phase1 = np.angle(yf1[fundamental_freq_index])
phase2 = np.angle(yf2[fundamental_freq_index])
phase_diff = phase1 - phase2

# 位相差を時間に変換
time_difference = phase_diff / (2 * np.pi * frequency)
print("signal2がsignal1に対して遅れている時間:", time_difference)
