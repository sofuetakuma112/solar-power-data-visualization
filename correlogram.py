import datetime
from es import fetch
import sys
from utils.q import calcQ
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import copy
import math
from utils.correlogram import (
    testEqualityDeltaBetweenDts,
    loadQAndDtForPeriod,
    unifyDeltasBetweenDts,
)

# > python3 correlogram.py 2022/04/01 2022/04/08 7.5 7
def main():
    args = sys.argv

    fromDtStr = args[1].split("/")
    toDtStr = args[2].split("/")
    fixedDaysLen = float(args[3])
    dynamicDaysLen = float(args[4])

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

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
    q_calc_end_dt = dt_all[0] + datetime.timedelta(days=dynamicDaysLen)
    q_calc_end_dt_index = 0
    for i, dt_crr in enumerate(dt_all):
        if dt_crr > q_calc_end_dt:
            q_calc_end_dt_index = i
            break
    print(f"dt_all列の先頭の日時から{dynamicDaysLen}日後の日時: {dt_all[q_calc_end_dt_index]}")

    dtStartLag_float, dtStartLag_int = math.modf((fixedDaysLen - dynamicDaysLen) / 2)

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    # 相互コレログラムを計算する際、計算値を{(fixedDaysLen - dynamicDaysLen) * 24 / 2}時間({(fixedDaysLen - dynamicDaysLen) / 2}日)シフトさせたタイミングで計算値と実測値の時系列データのズレが消える
    dts_q_calc_all = dt_all[:q_calc_end_dt_index]
    dts_q_calc_all_with_lag = list(
        map(
            lambda dt: dt
            + datetime.timedelta(days=dtStartLag_int)
            + datetime.timedelta(hours=dtStartLag_float * 24),
            dts_q_calc_all,
        )
    )

    print(
        f"dts_q_calc_all_with_{(fixedDaysLen - dynamicDaysLen) * 24 / 2}hours_delay[0]: {dts_q_calc_all_with_lag[0]}"
    )
    print(
        f"dts_q_calc_all_with_{(fixedDaysLen - dynamicDaysLen) * 24 / 2}hours_delay[-1]: {dts_q_calc_all_with_lag[-1]}"
    )
    Q_calc_all_applied_lag = list(
        map(
            lambda dt: max(calcQ(dt, 33.82794, 132.75093), 0) / 1000,
            dts_q_calc_all_with_lag,
        )
    )

    corr = np.correlate(Q_all, Q_calc_all_applied_lag)

    print(f"{corr.argmax()}秒スライドさせたとき相互相関が最大")  # corr.argmax()秒スライドさせた時が相互相関が最大
    largest_lag_sec = 6 * 60 * 60 - corr.argmax()
    print(f"真の計算値の時間 - 実測値の時間: {largest_lag_sec}")

    # axes = [plt.subplots()[1] for i in range(2)]
    axes = [plt.subplots() for _ in range(2)]

    # axes[0].plot(dt_all, Q_all, label="実測値(補完)")  # 補完データをプロット
    axes[0][1].plot(dt_all_copy, Q_all_copy, label="実測値", linestyle="dashed")

    print(int(largest_lag_sec))
    slided_dts_with_largest_lag_sec = list(
        map(
            lambda dt: dt + datetime.timedelta(seconds=int(largest_lag_sec)),
            dts_q_calc_all,
        )
    )
    axes[0][1].plot(
        dts_q_calc_all,
        list(
            map(
                lambda dt: max(calcQ(dt, 33.82794, 132.75093), 0) / 1000,
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
    axes[1][0].legend()
    plt.show()


if __name__ == "__main__":
    main()
