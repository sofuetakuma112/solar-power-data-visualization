import sys
import japanize_matplotlib
import matplotlib.pyplot as plt
import csv

# > python3 plot_corr_avg_and_lag.py data/csv/fixed/7.5,7.0,1.0.csv
if __name__ == "__main__":
    args = sys.argv
    file_path = args[1]

    rows = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    axes = [plt.subplots() for _ in range(1)]

    corrs = list(map(lambda row: float(row[3]), rows))
    lags = list(map(lambda row: float(row[4]), rows))

    xmin, xmax = -3, 3
    axes[0][1].scatter(corrs, lags, label="実測値")
    axes[0][1].set_xlabel("相互相関の最大値 / 計算値データ列のリスト長", fontsize=14)
    axes[0][1].set_ylabel("相互相関の最大値に対応するラグ[s]", fontsize=14)
    axes[0][1].hlines([0], min(corrs), max(corrs), linestyles="dashed")

    plt.show()
