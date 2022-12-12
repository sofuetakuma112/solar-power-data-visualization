import pickle
import datetime
from utils.es.fetch import fetchDocsByPeriod
from utils.file import get_pickle_file_path_by_datetime


if __name__ == "__main__":
    fromDt = datetime.datetime(2022, 3, 18)
    toDt = datetime.datetime.now() + datetime.timedelta(days=-2)

    fetchDocsByPeriod(fromDt, toDt)

    dtDiff = toDt - fromDt
    dt_crr = fromDt
    no_docs_dts = []
    for _ in range(dtDiff.days + 1):
        filePath = get_pickle_file_path_by_datetime(dt_crr)

        with open(filePath, "rb") as f:
            docs = pickle.load(f)
        if len(docs) < 100:
            no_docs_dts.append(dt_crr)

        dt_crr = dt_crr + datetime.timedelta(days=1)

    for dt in no_docs_dts:
        print(dt)

    start = fromDt
    for no_docs_dt in no_docs_dts:
        end = no_docs_dt
        diff = end - start

        if diff.days != 0:
            print(
                f"{format(start, '%Y/%m/%d')} ~ {format(end + datetime.timedelta(days=-1), '%Y/%m/%d')}, {diff.days}日間"
            )

        start = no_docs_dt + datetime.timedelta(days=1)
