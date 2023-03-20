import matplotlib.pyplot as plt

default_figsize = plt.rcParams["figure.figsize"]
default_dpi = plt.rcParams["figure.dpi"]

print(f"Default figsize: {default_figsize}")
print(f"Default dpi: {default_dpi}")

# フォントサイズを設定
font_size = 14

# 軸、軸メモリ、タイトル、判例のフォントサイズを変更
plt.rcParams["axes.labelsize"] = font_size
plt.rcParams["xtick.labelsize"] = font_size
plt.rcParams["ytick.labelsize"] = font_size
plt.rcParams["axes.titlesize"] = font_size
plt.rcParams["legend.fontsize"] = font_size

# サンプルデータを用意
x = [0, 1, 2, 3, 4]
y = [0, 2, 4, 6, 8]

# プロット
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, label="Sample Data")

# 軸ラベルとタイトルを設定
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_title("Custom Font Sizes")

# 判例を表示
ax.legend()

# グラフを表示
plt.show()
