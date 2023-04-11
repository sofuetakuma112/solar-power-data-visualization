import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# サンプリング周波数
fs = 1000  # サンプリング周波数（Hz）

# 時間軸データ
t = np.arange(0, 1, 1/fs)  # 1秒間のデータ

# 合成信号（例として、5Hzと50Hzの正弦波を使用）
x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)

# スペクトル解析（パワースペクトル密度を求める）
f, Pxx_den = signal.periodogram(x, fs)

# プロット
plt.semilogy(f, Pxx_den)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power spectral density")
plt.title("Spectral Analysis")
plt.grid(True)
plt.show()
