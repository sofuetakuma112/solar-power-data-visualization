import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from correlogram import loadQAndDtForAGivenPeriod

if __name__ == "__main__":
    args = sys.argv

    fromDtStr = args[1].split("/")
    toDtStr = args[2].split("/")

    fromDt = datetime.datetime(int(fromDtStr[0]), int(fromDtStr[1]), int(fromDtStr[2]))
    toDt = datetime.datetime(int(toDtStr[0]), int(toDtStr[1]), int(toDtStr[2]))

    dt_all, Q_all = loadQAndDtForAGivenPeriod(fromDt, toDt)

    print(f"dt_all[-1]: #{dt_all[-1]}")

    axes = [plt.subplots()[1] for i in range(1)]

    axes[0].plot(dt_all, Q_all, label="実測値")  # 実データをプロット
    axes[0].set_xlabel("日時")
    axes[0].set_ylabel("日射量[kW/m^2]")
    plt.show()