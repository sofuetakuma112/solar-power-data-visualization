import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# サンプルデータ生成
np.random.seed(0)
timestamps = pd.date_range('2022-01-01', periods=50, freq='H')
data = np.cumsum(np.random.randn(50))
time_series = pd.Series(data, index=timestamps)

# 傾きのしきい値
threshold = 0.5

# 傾きを計算
slope = np.diff(data)

# グループ化のためのラベル
group_label = 0
group_labels = [group_label]

# 傾きが似ている部分のグルーピング
for i in range(1, len(slope)):
    if abs(slope[i] - slope[i-1]) > threshold:
        group_label += 1
    group_labels.append(group_label)

# 各グループの始点と終点を列挙
group_labels = pd.Series(group_labels, index=timestamps[:-1])
groups = group_labels.groupby(group_labels)
for group, group_indices in groups:
    start = group_indices.index[0]
    end = group_indices.index[-1]
    print(f"Group {group}: Start = {start}, End = {end}")

# カラーマップを取得
cmap = plt.get_cmap("tab20")

# グループごとに色を変えてプロット
for group, group_indices in groups:
    start = group_indices.index[0]
    end = group_indices.index[-1] + pd.Timedelta(hours=1)
    plt.plot(time_series[start:end], label=f"Group {group}", color=cmap(group))

plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

