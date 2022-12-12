from calendar import c
import sys
import datetime
import math
import copy

from operator import itemgetter
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np


from utils.correlogram import (
    testEqualityDeltaBetweenDts,
    unifyDeltasBetweenDts,
    calc_ratios,
)
from utils.es.load import load_q_and_dt_for_period
from utils.es.fetch import fetchDocsByPeriod
from utils.q import calc_q_kw

# > python3 plot_filtered_q_by_ratio.py 2022/04/01 2.5 2
if __name__ == "__main__":
    args = sys.argv

    fromDtStr = args[1].split("/")
    fixedDaysLen = float(args[2])
    dynamicDaysLen = float(args[3])

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = fromDt + datetime.timedelta(days=math.ceil(fixedDaysLen))

    fetchDocsByPeriod(fromDt, toDt)  # pickleファイルがない場合、取得する
    dt_all, Q_all = load_q_and_dt_for_period(
        fromDt, fixedDaysLen
    )  # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)
    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)  # 時系列データのデルタを均一にする
    testEqualityDeltaBetweenDts(dt_all)  # 時系列データの点間が全て1.0[s]かテストする

    Q_all_copy = copy.deepcopy(Q_all)

    Q_calc_all = list(
        map(
            calc_q_kw,
            dt_all,
        )
    )

    ratios = calc_ratios(dt_all, Q_all)
    diffs_between_ratio_and_one = [
        (i, np.abs(1 - ratio)) for i, ratio in enumerate(ratios)
    ]
    total_len = len(diffs_between_ratio_and_one)

    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    axes = [plt.subplots() for _ in range(len(rates))]
    for i, rate in enumerate(rates):  # 全体のデータの何割を残すか
        last_idx = int(total_len * rate)
        mask = list(
            map(
                lambda idx_and_diff_ratio: idx_and_diff_ratio[0],  # 日時列のインデックスを取得する
                sorted(diffs_between_ratio_and_one, key=lambda x: x[1])[:last_idx],
            )
        )
        mask = sorted(mask)  # 日時列のインデックスを昇順で並び替える
        Q_all = copy.deepcopy(Q_all_copy)

        Q_filtered = itemgetter(*mask)(Q_all)
        dt_filtered = itemgetter(*mask)(dt_all)

        axes[i][1].set_xlabel("日時", fontsize=18)
        axes[i][1].set_ylabel("日射量", fontsize=18)
        axes[i][1].set_title(f"{rate * 100}%", fontsize=18)
        axes[i][1].scatter(dt_filtered, Q_filtered, label="実測値(フィルタリング済み)", s=1)
        axes[i][1].plot(dt_all, Q_calc_all, label="計算値", c="orange")
        axes[i][1].tick_params(axis="x", labelsize=18)
        axes[i][1].tick_params(axis="y", labelsize=18)
        axes[i][0].legend(fontsize=18)

    plt.show()
