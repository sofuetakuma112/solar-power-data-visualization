import datetime
from operator import itemgetter
from utils.es import fetch
from utils.file import getPickleFilePathByDatetime
import pickle
from utils.es.util import sortDocsByKey, isoformats2dt, extractFieldsFromDocs
from itertools import chain
import math
import os


class NotEnoughDocErr:
    def __init__(self):
        self.message = "Elasticsearchにデータが無い"


def loadQAndDtForPeriod(
    startDt,
    span,
    no_missing_data_err=False,  # Trueの場合、指定した期間にデータが取得できていない日があっても0埋めしてエラーを握りつぶす
):
    Q_all = []
    dt_all = []
    dt_crr_fetching = startDt

    span_float, span_int = math.modf(span)
    loopCount = 0
    while True:
        fetch.fetchDocsByDatetime(dt_crr_fetching)

        # エラーを握りつぶしてOKでかつ既に補完済みのデータをダンプしたファイルが存在する場合は、それを読み込む
        complemented_file_path = getPickleFilePathByDatetime(dt_crr_fetching, "raw")
        is_complemented_file_exist = os.path.isfile(complemented_file_path)
        if no_missing_data_err and is_complemented_file_exist:
            with open(complemented_file_path, "rb") as f:
                docs = pickle.load(f)
        else:
            raw_file_path = getPickleFilePathByDatetime(dt_crr_fetching, "raw")
            with open(raw_file_path, "rb") as f:
                docs = pickle.load(f)
            if len(docs) < 100:
                if no_missing_data_err:
                    # docsをループで回して年/月/日/時/分/秒レベルでデータが無いものを0扱いで補完する
                    def _complement_docs(dt_crr_complementing, end_dt, docs):
                        while dt_crr_complementing < end_dt:
                            # dt_crr_complementing_utc = (
                            #     dt_crr_complementing + datetime.timedelta(hours=-9)
                            # )
                            docs.append(
                                {
                                    "_source": {
                                        "JPtime": (dt_crr_complementing.isoformat()),
                                        "solarIrradiance(kw/m^2)": 0,
                                    }
                                }
                            )
                            dt_crr_complementing += datetime.timedelta(seconds=1)

                        return docs

                    if len(docs) == 0:
                        # 全くデータが無い
                        end_dt = dt_crr_fetching + datetime.timedelta(days=1)
                        dt_crr_complementing = dt_crr_fetching
                        docs = _complement_docs(dt_crr_complementing, end_dt, docs)
                    else:
                        # 途中で計測できなくなっている
                        end_dt = datetime.datetime(
                            dt_crr_fetching.year,
                            dt_crr_fetching.month,
                            dt_crr_fetching.day,
                            0,
                            0,
                            0,
                            0,
                        ) + datetime.timedelta(days=1)
                        dt_crr_complementing = datetime.datetime(
                            dt_crr_fetching.year,
                            dt_crr_fetching.month,
                            dt_crr_fetching.day,
                            dt_crr_fetching.hour,
                            dt_crr_fetching.minute,
                            dt_crr_fetching.second,
                            0,
                        ) + datetime.timedelta(seconds=1)
                        len_docs_before_complement = len(docs)
                        docs = _complement_docs(dt_crr_complementing, end_dt, docs)
                        print(f"len(docs): {len(docs)}")
                        print(
                            f"docs[len_docs_before_complement - 1]: {docs[len_docs_before_complement - 1]}"
                        )
                        print(
                            f"docs[len_docs_before_complement]: {docs[len_docs_before_complement]}"
                        )
                        print(f"docs[-1]: {docs[-1]}")
                else:
                    return [NotEnoughDocErr(), None]

        # この時点で、補完済み or 補完する必要のないデータのみ
        # 補完済みデータとしてダンプしておく
        if not is_complemented_file_exist:
            with open(complemented_file_path, "wb") as f:
                pickle.dump(docs, f)

        docs = sortDocsByKey(docs, "JPtime")
        jptimes = extractFieldsFromDocs(docs, "JPtime")
        dts_per_day = isoformats2dt(jptimes)

        # print(f"dt_crr_fetching: {dt_crr_fetching}")
        # print(f"len(dts_per_day): {len(dts_per_day)}")
        # if len(dts_per_day) < 86400:
        #     print(f"データが補完の有無に関わらず86400点なかったもの: {dt_crr_fetching}")
        # print(f"dts_per_day[0]: {dts_per_day[0]}")
        # print(f"dts_per_day[-1]: {dts_per_day[-1]}")

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
            raise ValueError(f"データがない日が含まれている: {dt_crr_fetching}")

        dt_crr_fetching = dt_crr_fetching + datetime.timedelta(days=1)
        loopCount += 1

    return [dt_all, Q_all]


def loadQAndDtForAGivenPeriod(fromDt, toDt, includesNoDataDay=False):
    Q_all = []
    dt_all = []
    dtDiff = toDt - fromDt
    dt_crr = fromDt
    for _ in range(dtDiff.days + 1):
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
