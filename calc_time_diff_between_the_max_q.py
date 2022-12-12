import datetime
from utils.es import fetch
from utils.q import calc_q_kw
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
from utils.correlogram import (
    NotEnoughLengthErr,
    testEqualityDeltaBetweenDts,
    unifyDeltasBetweenDts,
)
from utils.es.load import load_q_and_dt_for_period
import argparse
from utils.q import calc_q_kw

# > python3 calc_time_diff_between_the_max_q.py -d 2022/04/01
def main():
    parser = argparse.ArgumentParser(description="add two integer")
    parser.add_argument("-d", type=str)  # 日付
    args = parser.parse_args()

    fromDtStr = args.d.split("/")
    days_len = 1.0

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = fromDt + datetime.timedelta(days=1)

    fetch.fetchDocsByPeriod(fromDt, toDt)  # pickleファイルがない場合、取得する

    # 与えた期間の日射量と計測日時をファイルから読み込む(dtでソート済み)
    dt_all, Q_all = load_q_and_dt_for_period(fromDt, days_len, True)

    print(f"dt_all[0]: {dt_all[0]}")
    print(f"dt_all[-1]: {dt_all[-1]}")

    # 時系列データのデルタを均一にする
    dt_all, Q_all = unifyDeltasBetweenDts(dt_all, Q_all)

    # 時系列データの点間が全て1.0[s]かテストする
    testEqualityDeltaBetweenDts(dt_all)

    dt_all = np.array(dt_all)
    Q_all = np.array(Q_all)
    q_calc_all = np.vectorize(calc_q_kw)(dt_all)

    print(f"len(dt_all): {len(dt_all)}")
    print(f"len(Q_all): {len(Q_all)}")
    print(f"len(q_calc_all): {len(q_calc_all)}")

    corr = np.correlate(Q_all, q_calc_all, "full")

    estimated_delay = corr.argmax() - (len(q_calc_all) - 1)
    print("estimated delay is " + str(estimated_delay))

    print(abs(dt_all[Q_all.argmax()] - dt_all[q_calc_all.argmax()]))

    axes = [plt.subplots() for _ in range(1)]
    axes[0][1].plot(
        dt_all,
        Q_all,
        label="実測値",
    )
    axes[0][1].plot(
        dt_all,
        q_calc_all,
        label="計算値",
    )
    axes[0][1].set_xlabel("日時")
    axes[0][1].set_ylabel("日射量[kW/m^2]")
    plt.show()


if __name__ == "__main__":
    main()
