import datetime
from utils.es import fetch
import sys
from utils.q import calc_q_kw
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import copy
from utils.correlogram import (
    NotEnoughLengthErr,
    testEqualityDeltaBetweenDts,
    unifyDeltasBetweenDts,
    calc_dts_for_q_calc,
    slides_q_calc_for_corr,
)
from utils.es.load import loadQAndDtForPeriod
import argparse
import math

# > python3 correlogram.py 2022/04/01 2022/04/08 7.5 7
def main():
    parser = argparse.ArgumentParser(description="add two integer")
    parser.add_argument("-f", type=str)  # from date str format
    parser.add_argument("-fd", type=float, default=2.5)  # fixed day length
    parser.add_argument("-dd", type=float, default=2.0)  # dynamic day length
    parser.add_argument("-p", type=float, default=1.0)  # percentage of data
    parser.add_argument(
        "-tp", type=int, default=1.0
    )  # 相互相関を計算する際に理論値の波形に近い波形を上位何日使用するか
    parser.add_argument("-rz", "--replace_zero", action="store_true")
    parser.add_argument("-sl", "--should_log", action="store_true")  # ログを取るか
    args = parser.parse_args()

    fromDtStr = args.f.split("/")
    fixedDaysLen = float(args.fd)
    dynamicDaysLen = float(args.dd)

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = fromDt + datetime.timedelta(days=math.ceil(fixedDaysLen))

    fetch.fetchDocsByPeriod(fromDt, toDt)  # pickleファイルがない場合、取得する

    # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)
    dt_all, Q_all = loadQAndDtForPeriod(fromDt, fixedDaysLen)
    dt_all_copy = copy.deepcopy(dt_all)  # 補完が正しく行えているか確認する用
    Q_all_copy = copy.deepcopy(Q_all)  # 補完が正しく行えているか確認する用

    print(f"dt_all[0]: {dt_all[0]}")
    print(f"dt_all[-1]: {dt_all[-1]}")

    # 時系列データのデルタを均一にする
    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)

    # 時系列データの点間が全て1.0[s]かテストする
    testEqualityDeltaBetweenDts(dt_all)

    # 実測値の日時データからトリムして計算値用の日時データを作るので
    # トリムする範囲を指定するためのインデックスを求める
    dts_for_q_calc_or_err = calc_dts_for_q_calc(dt_all, dynamicDaysLen)

    dts_for_q_calc = []
    if isinstance(dts_for_q_calc_or_err, NotEnoughLengthErr):
        return dts_for_q_calc_or_err, None
    else:
        dts_for_q_calc = dts_for_q_calc_or_err

    print(f"dt_all列の先頭の日時から{dynamicDaysLen}日後の日時: {dts_for_q_calc[-1]}")

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    # 相互コレログラムを計算する際、計算値を{(fixedDaysLen - dynamicDaysLen) * 24 / 2}時間({(fixedDaysLen - dynamicDaysLen) / 2}日)シフトさせたタイミングで計算値と実測値の時系列データのズレが消える
    Q_calc_all_applied_lag = slides_q_calc_for_corr(
        dts_for_q_calc, fixedDaysLen, dynamicDaysLen
    )

    corr = np.correlate(Q_all, Q_calc_all_applied_lag)

    print(f"{corr.argmax()}秒スライドさせたとき相互相関が最大")  # corr.argmax()秒スライドさせた時が相互相関が最大
    largest_lag_sec = 6 * 60 * 60 - corr.argmax()
    print(f"真の計算値の時間 - 実測値の時間: {largest_lag_sec}")

    print(f"相互相関の最大値 / 計算値のデータ列の長さ: {corr.max() / len(Q_calc_all_applied_lag)}")

    # axes = [plt.subplots()[1] for i in range(2)]
    axes = [plt.subplots() for _ in range(2)]

    # axes[0].plot(dt_all, Q_all, label="実測値(補完)")  # 補完データをプロット
    axes[0][1].plot(dt_all_copy, Q_all_copy, label="実測値", linestyle="dashed")

    print(int(largest_lag_sec))
    slided_dts_with_largest_lag_sec = list(
        map(
            lambda dt: dt + datetime.timedelta(seconds=int(largest_lag_sec)),
            dts_for_q_calc,
        )
    )
    axes[0][1].plot(
        dts_for_q_calc,
        list(
            map(
                calc_q_kw,
                slided_dts_with_largest_lag_sec,
            )
        ),
        label="計算値(相互相関が最大となるラグを適用)",
        linestyle="dashed",
    )

    axes[0][1].set_xlabel("日時", fontsize=20)
    axes[0][1].set_ylabel("日射量[kW/m^2]", fontsize=20)

    print(f"len(corr): {len(corr)}")

    axes[1][1].set_xlabel("実測値の日時 - 計算値の日時[s]")
    axes[1][1].set_ylabel("相互相関")
    axes[1][1].plot(
        [
            i - (fixedDaysLen - dynamicDaysLen) * 24 * 60 * 60 / 2
            for i in range(len(corr))
        ],
        corr,
        color="r",
    )
    axes[0][1].tick_params(axis="x", labelsize=20)
    axes[0][1].tick_params(axis="y", labelsize=20)

    axes[0][0].legend(fontsize=20)
    plt.show()


if __name__ == "__main__":
    main()
