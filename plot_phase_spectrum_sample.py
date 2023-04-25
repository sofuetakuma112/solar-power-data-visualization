import numpy as np
import matplotlib.pyplot as plt

# テストデータの作成
t = np.linspace(0, 1, 1000)
x = 2 * np.pi * 10 * t

# FFTを実行して、周波数スペクトルと位相スペクトルを取得する
y = np.fft.fft(np.sin(x))
freqs = np.fft.fftfreq(len(y))
phases = np.angle(y)

# 位相スペクトルをプロットする
plt.plot(freqs, phases)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [rad]")
plt.show()
