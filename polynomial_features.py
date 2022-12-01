# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures

# np.random.seed(0)

# x = np.linspace(-5, 5, 20)
# y = (
#     2 * x
#     - 2 * (x**2)
#     + 0.5 * (x**3)
#     + np.random.normal(loc=0, scale=10, size=len(x))
# )
# x = x[:, np.newaxis]
# y = y[:, np.newaxis]
# # 2次元の特徴量に変換
# polynomial_features = PolynomialFeatures(degree=2)
# x_poly = polynomial_features.fit_transform(x)

# # y = b0 + b1x + b2x^2 の b0～b2 を算出
# model = LinearRegression()
# model.fit(x_poly, y)
# y_pred = model.predict(x_poly)

# # 評価
# rmse = np.sqrt(mean_squared_error(y, y_pred))
# r2 = r2_score(y, y_pred)
# print(f"rmse : {rmse}")
# print(f"R2 : {r2}")

# # 可視化
# plt.scatter(x, y)
# plt.plot(x, y_pred, color="r")
# plt.show()

import datetime
from functools import reduce
import os
from utils.es.fetch import fetchDocsByPeriod
import numpy as np
import japanize_matplotlib
import math
from utils.correlogram import (
    NotEnoughLengthErr,
    testEqualityDeltaBetweenDts,
    unifyDeltasBetweenDts,
    calc_dts_for_q_calc,
    slides_q_calc_for_corr,
    calc_ratios,
)
from utils.es.load import NotEnoughDocErr, loadQAndDtForPeriod
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils.q import calc_q_kw

if __name__ == "__main__":

    def main():
        parser = argparse.ArgumentParser(description="add two integer")
        parser.add_argument("-f", type=str)  # from date str format
        parser.add_argument("-t", type=str)  # from date str format
        parser.add_argument("-fd", type=float, default=2.5)  # fixed day length
        parser.add_argument("-dd", type=float, default=2.0)  # dynamic day length
        # parser.add_argument("-t", type=float, default=0.3)  # threshold
        # parser.add_argument("-p", type=float, default=1.0)  # percentage of data
        # parser.add_argument("-md", "--mode_dynamic", action="store_true")
        # parser.add_argument("-ai", "--auto_increment", action="store_true")
        # parser.add_argument("-rz", "--replace_zero", action="store_true")
        # parser.add_argument("-sl", "--should_log", action="store_true")
        args = parser.parse_args()

        fromDtStr = args.f.split("/")
        fromDt = datetime.datetime(
            int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2])
        )
        toDtStr = args.t.split("/")
        toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

        fixedDaysLen = float(args.fd)
        dynamicDaysLen = float(args.dd)

        fetchDocsByPeriod(fromDt, toDt)  # pickleファイルがない場合、取得する

        dt_all_or_err, Q_all = loadQAndDtForPeriod(
            fromDt, fixedDaysLen
        )  # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)
        dt_all = None
        if isinstance(dt_all_or_err, NotEnoughDocErr):
            return dt_all_or_err, None
        else:
            dt_all = dt_all_or_err

        dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)  # 時系列データのデルタを均一にする
        testEqualityDeltaBetweenDts(dt_all)  # 時系列データの点間が全て1.0[s]かテストする

        Qs_calc = list(
            map(
                calc_q_kw,
                dt_all,
            )
        )

        # (実測値 / 理論値)を各日時ごとに計算して、ソートして上から何割かだけの日射量を採用して残りは0にする
        ratios = calc_ratios(dt_all, Q_all)
        diffs_between_ratio_and_one = [  # 比が1からどれだけ離れているか
            (i, np.abs(1 - ratio)) for i, ratio in enumerate(ratios)
        ]

        # 指定した範囲ごとに範囲に含まれるデータ点を数え上げて一番多いところ？を採用する？
        def _update(hist, ratio):
            bins = np.linspace(0, 1, 100)
            space = bins[1] - bins[0]

            digitized = np.digitize(ratio[-1], bins)

            min_digit = 0
            max_digit = len(bins)

            # keyName = f"{(digitized - 1) * space:.2f}-{digitized * space:.2f}"
            if digitized == min_digit:
                keyName = bins[-1]
            elif digitized == max_digit:
                keyName = -float("inf")
            else:
                keyName = float(f"{(digitized - 1) * space:.2f}")

            if keyName in hist:
                hist[keyName].append(ratio)
            else:
                hist[keyName] = [ratio]

            return hist

        hist = reduce(lambda hist, t: _update(hist, t), diffs_between_ratio_and_one, {})

        for k in hist:
            print(f"key: {k}, 範囲に含まれるデータ点の数: {len(hist[k])}")

        # fig, axes = plt.subplots(2, 2, tight_layout=True)
        # axes[0][0].plot(
        #     [dt_all[t[0]] for t in diffs_between_ratio_and_one],
        #     [t[1] for t in diffs_between_ratio_and_one],
        # )
        # axes[0][1].plot(
        #     dt_all,
        #     Q_all,
        # )
        # axes[0][1].plot(
        #     dt_all,
        #     Qs_calc,
        # )

        # axes[1][0].hist([t[1] for t in diffs_between_ratio_and_one])

        # plt.show()

        # max_index = np.argmax([t[1] for t in diffs_between_ratio_and_one])
        # min_index = np.argmin([t[1] for t in diffs_between_ratio_and_one])

        # print(f"max_ratio: {diffs_between_ratio_and_one[max_index][1]}")
        # print(f"min_ratio: {diffs_between_ratio_and_one[min_index][1]}")

        # print(
        #     f"max_ratio dt: {dt_all[max_index]}, max_ratio q: {Q_all[max_index]}, max_ratio calc_q: {Qs_calc[max_index]}"
        # )
        # print(
        #     f"min_ratio dt: {dt_all[min_index]}, min_ratio q: {Q_all[min_index]}, min_ratio calc_q: {Qs_calc[min_index]}"
        # )

    main()
