import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import loadQAndDtForAGivenPeriod
import os
import json
import numpy as np
from utils.q import calc_q_kw
from utils.correlogram import (
    unify_deltas_between_dts,
)

if __name__ == "__main__":
    json_open = open("data/json/sorted_diffs.json", "r")
    diff_and_dates = json.load(json_open)

    for diff_and_date in diff_and_dates:
        fromDt = datetime.datetime.strptime(diff_and_date[-1], "%Y-%m-%d")
        toDt = fromDt + datetime.timedelta(days=1)

        dt_all, Q_all = loadQAndDtForAGivenPeriod(fromDt, toDt, True)

        # 時系列データのデルタを均一にする
        dt_all, Q_all = unify_deltas_between_dts(dt_all, Q_all)

        dt_all = np.array(dt_all)
        Q_all = np.array(Q_all)
        q_calc_all = np.vectorize(calc_q_kw)(dt_all)

        corr = np.correlate(Q_all, q_calc_all, "full")
        estimated_delay = corr.argmax() - (len(q_calc_all) - 1)

        print(f"{fromDt}: {estimated_delay}")
