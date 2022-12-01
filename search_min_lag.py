import datetime
from utils.es import fetch
import sys
from utils.q import calc_q_kw
import numpy as np
import japanize_matplotlib
import math
from correlogram import unifyDeltasBetweenDts
import matplotlib.pyplot as plt
from utils.correlogram import calcLag, shiftDts
from utils.es.load import loadQAndDtForPeriod

def main():
    args = sys.argv

    fromDtStr = args[1].split("/")
    toDtStr = args[2].split("/")
    fixedDaysLen = float(args[3])
    dynamicDaysLen = float(args[4])
    type = int(args[5])

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

    fetch.fetchDocsByPeriod(fromDt, toDt)  # pickleファイルがない場合、取得する

    # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)
    dt_all, Q_all = loadQAndDtForPeriod(fromDt, fixedDaysLen)
    # 時系列データのデルタを均一にする
    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)

    # 実測値の日時データからトリムして計算値用の日時データを作るので
    # トリムする範囲を指定するためのインデックスを求める
    q_calc_end_dt = dt_all[0] + datetime.timedelta(days=dynamicDaysLen)
    q_calc_end_dt_index = 0
    for i, dt_crr in enumerate(dt_all):
        if dt_crr > q_calc_end_dt:
            q_calc_end_dt_index = i
            break

    dtStartLag_float, dtStartLag_int = math.modf((fixedDaysLen - dynamicDaysLen) / 2)

    # Q_calc_allの時系列データを実測値の時系列データより6時間進める
    # 相互コレログラムを計算する際、計算値を{(fixedDaysLen - dynamicDaysLen) * 24 / 2}時間({(fixedDaysLen - dynamicDaysLen) / 2}日)シフトさせたタイミングで計算値と実測値の時系列データのズレが消える
    dts_for_calc = dt_all[:q_calc_end_dt_index]

    # TODO: 以下ループ
    # 相互相関を求めるために計算値の日時を(スライド全量 / 2)だけずらす
    dts_for_calc_applied_lag = shiftDts(dts_for_calc, dtStartLag_int, dtStartLag_float)

    largest_lag_sec = calcLag(Q_all, dts_for_calc_applied_lag)  # 相互相関が最大となるラグを返す
    print(f"largest_lag_sec: {largest_lag_sec}")
    dts_applied_largest_lag_sec = list(
        map(
            lambda dt: dt + datetime.timedelta(seconds=int(largest_lag_sec)),
            dts_for_calc,
        )
    )

    # 相互相関が最大になったラグを適用
    dts_for_calc_applied_lag_and_half_slides = shiftDts(
        dts_applied_largest_lag_sec, dtStartLag_int, dtStartLag_float
    )

    # 1: 計算値を係数掛けした時の相互相関を計算して最大のラグを求める
    # FIXME: 一律で同じ係数を計算値の掛けても相互相関の結果は変化しないのでこの方法は無意味
    if type == 1:
        # coefs = list(map(lambda x: x / 10, range(1, 11)))
        coefs = [0.1, 0.7, 0.8, 0.9]
        axes = [plt.subplots() for _ in range(len(coefs))]

        lags_sec = []
        for i, coef in enumerate(
            coefs
        ):  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            Qs_calc = list(
                map(
                    lambda dt: calc_q_kw(dt) * coef,
                    # dts_applied_largest_lag_sec,
                    dts_for_calc,
                )
            )

            largest_lag_sec = calcLag(
                Q_all,
                # dts_for_calc_applied_lag_and_half_slides,
                shiftDts(dts_for_calc, dtStartLag_int, dtStartLag_float),
                coef,
            )

            axes[i][1].set_xlabel("日時", fontsize=14)
            axes[i][1].set_ylabel("日射量", fontsize=14)
            axes[i][1].plot(
                dt_all,
                Q_all,
                label="実測値",
            )
            axes[i][1].plot(
                # dts_for_calc,
                list(
                    map(
                        lambda dt: dt
                        + datetime.timedelta(seconds=int(largest_lag_sec)),
                        dts_for_calc,
                    )
                ),
                Qs_calc,
                label="計算値(相互相関が最大となるラグを適用)",
            )
            axes[i][1].tick_params(axis="x", labelsize=14)
            axes[i][1].tick_params(axis="y", labelsize=14)
            axes[i][0].legend(fontsize=14)

            print(f"coef: {coef}, largest_lag_sec: {largest_lag_sec}")
            lags_sec.append(largest_lag_sec)
        print(f"min(lags_sec): {min(lags_sec)}")
        
        plt.show()

    # 2: 実測値をラグを適用した計算値の各日時ごとの日射量比を求める
    if type == 2:
        ratios = []
        Qs_calc = list(
            map(
                calc_q_kw,
                dts_applied_largest_lag_sec,
            )
        )
        print(f"len(Q_all): {len(Q_all)}")
        print(f"len(Qs_calc): {len(Qs_calc)}")
        for q, q_calc in zip(
            Q_all[:q_calc_end_dt_index],
            Qs_calc,
        ):
            # if q_calc == 0:
            #     ratio = q / 1e-6
            # else:
            #     ratio = q / q_calc
            # ratios.append(ratio)

            ratio = q_calc - q if abs(q_calc - q) < 0.2 else 0
            ratios.append(ratio)

        mask = np.where(np.array(ratios) != 0)[0]

        axes = [plt.subplots() for _ in range(3)]
        axes[0][1].set_xlabel("日時")
        axes[0][1].set_ylabel("計算値(ラグ適用済み) - 実測値")
        axes[0][1].plot(
            dt_all[:q_calc_end_dt_index],
            ratios,
        )

        axes[1][1].set_xlabel("日時")
        axes[1][1].set_ylabel("日射量")
        axes[1][1].plot(
            dt_all,
            Q_all,
            label="実測値",
        )
        axes[1][1].plot(
            dt_all[:q_calc_end_dt_index],
            Qs_calc,
            label="計算値(相互相関が最大となるラグを適用)",
        )
        axes[1][0].legend()

        axes[2][1].set_xlabel("日時")
        axes[2][1].set_ylabel("日射量")
        axes[2][1].scatter(
            np.array(dt_all)[:q_calc_end_dt_index][mask],
            np.array(Q_all)[:q_calc_end_dt_index][mask],
            label="実測値",
            s=1,
        )
        # axes[2][1].scatter(
        #     np.array(dt_all)[:q_calc_end_dt_index][mask],
        #     np.array(Qs_calc)[mask],
        #     label="計算値(相互相関が最大となるラグを適用)",
        # )

        plt.show()


if __name__ == "__main__":
    main()
