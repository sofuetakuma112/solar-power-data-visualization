import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import loadQAndDtForAGivenPeriod

# > python3 plotQOnGivenSpan.py 2022/04/01 00:00:00 2022/05/01 00:00:00
if __name__ == "__main__":
    args = sys.argv

    fromDtTillDay_str = args[1].split("/")
    fromDtTillSec_str = args[2].split(":")
    toDtDtTillDay_str = args[3].split("/")
    toDtTillSec_str = args[4].split(":")

    fromDt = datetime.datetime(
        int(fromDtTillDay_str[0]),
        int(fromDtTillDay_str[1]),
        int(fromDtTillDay_str[2]),
        int(fromDtTillSec_str[0]),
        int(fromDtTillSec_str[1]),
        int(fromDtTillSec_str[2]),
    )

    toDt = datetime.datetime(
        int(toDtDtTillDay_str[0]),
        int(toDtDtTillDay_str[1]),
        int(toDtDtTillDay_str[2]),
        int(toDtTillSec_str[0]),
        int(toDtTillSec_str[1]),
        int(toDtTillSec_str[2]),
    )

    dt_all, Q_all = loadQAndDtForAGivenPeriod(fromDt, toDt, True)

    axes = [plt.subplots()[1] for i in range(1)]

    axes[0].plot(dt_all, Q_all, label="実測値")  # 実データをプロット
    axes[0].set_xlabel("日時")
    axes[0].set_ylabel("日射量[kW/m^2]")
    axes[0].set_xlim(fromDt, toDt)
    plt.show()
