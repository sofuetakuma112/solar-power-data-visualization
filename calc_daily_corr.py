import datetime
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.es.load import loadQAndDtForAGivenPeriod
import os
import json
import numpy as np
from utils.q import calc_q_kw

if __name__ == "__main__":
    json_open = open("data/json/sorted_diffs.json", "r")
    diff_and_dates = json.load(json_open)

    for diff_and_date in diff_and_dates:
        fromDt = datetime.datetime.strptime(diff_and_date[-1], "%Y-%m-%d")
        toDt = fromDt + datetime.timedelta(days=1)

        dt_all, Q_all = loadQAndDtForAGivenPeriod(fromDt, toDt, True)

        dt_all = np.array(dt_all)
        Q_all = np.array(Q_all)

        corr = np.correlate(Q_all, np.vectorize(calc_q_kw)(dt_all), "full")
        estimated_delay = corr.argmax() - (len(Q_all) - 1)

        print(f"{fromDt}: {estimated_delay}")
