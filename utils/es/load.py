import datetime
from operator import itemgetter
from utils.es import fetch
from utils.file import getPickleFilePathByDatetime
import pickle
from utils.es.util import sortDocsByKey, isoformats2dt, extractFieldsFromDocs
from itertools import chain
import math

class NotEnoughDocErr:
    def __init__(self):
        self.message = "Elasticsearchにデータが無い"

def loadQAndDtForPeriod(startDt, span):
    Q_all = []
    dt_all = []
    dt_crr = startDt

    span_float, span_int = math.modf(span)
    loopCount = 0
    while True:
        fetch.fetchDocsByDatetime(dt_crr)
        filePath = getPickleFilePathByDatetime(dt_crr)
        with open(filePath, "rb") as f:
            docs = pickle.load(f)
        if len(docs) < 100:
            return [NotEnoughDocErr(), None]
        docs = sortDocsByKey(docs, "JPtime")
        jptimes = extractFieldsFromDocs(docs, "JPtime")
        dts_per_day = isoformats2dt(jptimes)
        if loopCount == 0:
            lastDt = (
                dts_per_day[0]  # dts_per_day[0]から{fixedDaysLen}日後をlastDtとする
                + datetime.timedelta(days=span_int)
                + datetime.timedelta(hours=span_float * 24)
            )
        Qs_per_day = extractFieldsFromDocs(docs, "solarIrradiance(kw/m^2)")

        if all(list(map(lambda dt: dt > lastDt, dts_per_day))):
            # jptimesの全ての日時がlastDtより未来の日時
            indexes = [i for i, _ in enumerate(Q_all) if dt_all[i] <= lastDt]
            Q_all = itemgetter(*indexes)(Q_all)
            dt_all = itemgetter(*indexes)(dt_all)
            break

        Q_all = list(chain(Q_all, Qs_per_day))
        dt_all = list(chain(dt_all, dts_per_day))

        if len(Qs_per_day) < 10:
            raise ValueError(f"データがない日が含まれている: {dt_crr}")

        dt_crr = dt_crr + datetime.timedelta(days=1)
        loopCount += 1

    return [dt_all, Q_all]


def loadQAndDtForAGivenPeriod(fromDt, toDt, includesNoDataDay = False):
    Q_all = []
    dt_all = []
    dtDiff = toDt - fromDt
    dt_crr = fromDt
    for i in range(dtDiff.days + 1):
        fetch.fetchDocsByDatetime(dt_crr)
        filePath = getPickleFilePathByDatetime(dt_crr)
        with open(filePath, "rb") as f:
            docs = pickle.load(f)
        docs = sortDocsByKey(docs, "JPtime")
        jptimes = extractFieldsFromDocs(docs, "JPtime")
        dts_per_day = isoformats2dt(jptimes)
        Qs_per_day = extractFieldsFromDocs(docs, "solarIrradiance(kw/m^2)")

        Q_all = list(chain(Q_all, Qs_per_day))
        dt_all = list(chain(dt_all, dts_per_day))

        if len(Qs_per_day) < 10 and not includesNoDataDay:
            raise ValueError(f"データがない日が含まれている: {dt_crr}")

        dt_crr = dt_crr + datetime.timedelta(days=1)

    return [dt_all, Q_all]
