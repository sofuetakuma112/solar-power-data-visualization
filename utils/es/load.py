import datetime
import json
from utils.es import fetch
from utils.file import get_json_file_path_by_datetime
from utils.es.util import (
    extractFieldsFromDocs,
    extractFieldsFromDoc,
    isoformat2dt,
)
import math
import numpy as np

import logging


class NotEnoughDocErr:
    def __init__(self):
        self.message = "Elasticsearchにデータが無い"


def doc_to_dt(doc):
    return isoformat2dt(extractFieldsFromDoc(doc, "JPtime"))


def create_doc_dict(dt, q):
    return {
        "_source": {
            "JPtime": (dt.isoformat()),
            "solarIrradiance(kw/m^2)": q,
        }
    }


def time_to_seconds(t):
    return datetime.timedelta(
        hours=t.hour, minutes=t.minute, seconds=t.second
    ).total_seconds()


EPOCH_TIME = datetime.datetime(1970, 1, 1)

# データの補完はunify_deltas_between_dtsに任せる
# そのためにここでは00:00:00と23:59:59にそれぞれデータ点を挿入する
def load_q_and_dt_for_period(
    start_dt,
    span,
):
    q_all = np.array([])
    dt_all = np.array([])
    dt_crr_fetching = start_dt

    span_float, span_int = math.modf(span)
    is_first_loop = True

    for _ in range(int(np.ceil(span))):
        fetch.fetch_docs_by_datetime(dt_crr_fetching)

        json_file_path = get_json_file_path_by_datetime(dt_crr_fetching)
        with open(json_file_path, "rb") as f:
            docs = np.array(json.load(f))

        total_seconds_list = np.array([])
        if docs.size != 0:
            total_seconds_list = np.vectorize(
                lambda doc: (
                    isoformat2dt(extractFieldsFromDoc(doc, "JPtime")) - EPOCH_TIME
                ).total_seconds()
            )(docs)
            sorted_indexes = np.argsort(total_seconds_list)
            docs = docs[sorted_indexes]

        year = dt_crr_fetching.year
        month = dt_crr_fetching.month
        day = dt_crr_fetching.day
        date = datetime.datetime(year, month, day)

        if docs.size == 0:
            # 00:00:00 ~ 23:59:59まで全て補完する
            docs = np.vectorize(
                lambda second_from_start: create_doc_dict(
                    date + datetime.timedelta(seconds=int(second_from_start)), 0
                )
            )(np.arange(0, 86400, 1))
        else:
            first_dt = isoformat2dt(extractFieldsFromDoc(docs[0], "JPtime"))
            last_dt = isoformat2dt(extractFieldsFromDoc(docs[-1], "JPtime"))

            logging.debug(f"first_dt: {first_dt}")
            logging.debug(f"last_dt: {last_dt}")

            start_dt = datetime.datetime(
                first_dt.year, first_dt.month, first_dt.day, 0, 0, 0
            )
            end_dt = datetime.datetime(
                last_dt.year, last_dt.month, last_dt.day, 0, 0, 0
            ) + datetime.timedelta(days=1)

            logging.debug(f"start_dt: {start_dt}")
            logging.debug(f"end_dt: {end_dt}")

            diff_seconds_from_start = np.floor((first_dt - start_dt).seconds)

            logging.debug(f"diff_seconds_from_start: {diff_seconds_from_start}")

            if diff_seconds_from_start == 0:
                # docs_from_start_to_firstの長さは0
                docs_from_start_to_first = np.array([])
            else:
                docs_from_start_to_first = np.vectorize(
                    lambda second_from_start: create_doc_dict(
                        date + datetime.timedelta(seconds=second_from_start), 0
                    )
                )(np.arange(0, diff_seconds_from_start, 1))

            if docs_from_start_to_first.size > 0:
                logging.debug(f"left 0: {doc_to_dt(docs_from_start_to_first[0])}")
                logging.debug(f"left -1: {doc_to_dt(docs_from_start_to_first[-1])}")

                logging.debug(f"middle 0: {doc_to_dt(docs[0])}")
                logging.debug(f"middle -1: {doc_to_dt(docs[-1])}")

            # 2
            diff_seconds_from_last_to_end = np.floor((end_dt - last_dt).seconds)
            offset = np.ceil(
                time_to_seconds(
                    datetime.time(
                        last_dt.hour,
                        last_dt.minute,
                        last_dt.second,
                        last_dt.microsecond,
                    )
                )
            )
            if offset == offset + diff_seconds_from_last_to_end:
                docs_from_last_to_end = np.array([])
            else:
                docs_from_last_to_end = np.vectorize(
                    lambda second_from_start: create_doc_dict(
                        date + datetime.timedelta(seconds=second_from_start), 0
                    )
                )(
                    np.arange(offset + 1, offset + diff_seconds_from_last_to_end + 1, 1)
                )  # FIXME: +1しなくても良い方を探す

            if docs_from_last_to_end.size > 0:
                logging.debug(f"right 0: {doc_to_dt(docs_from_last_to_end[0])}")
                logging.debug(f"right -1: {doc_to_dt(docs_from_last_to_end[-1])}")

            docs = np.append(docs_from_start_to_first, docs)
            docs = np.append(docs, docs_from_last_to_end)

            logging.debug(f"doc_to_dt(docs[0]): {doc_to_dt(docs[0])}")
            logging.debug(f"doc_to_dt(docs[-1]): {doc_to_dt(docs[-1])}")
            logging.debug(
                f"diff_seconds_from_last_to_end: {diff_seconds_from_last_to_end}"
            )
            logging.debug(f"offset: {offset}", end="\n\n")

        # TODO: 時系列データが正しく並んでいるかテストする
        total_seconds_list = np.vectorize(
            lambda doc: (
                isoformat2dt(extractFieldsFromDoc(doc, "JPtime")) - EPOCH_TIME
            ).total_seconds()
        )(docs)
        source_indexes = np.argsort(total_seconds_list)
        target_indexes = np.arange(0, source_indexes.size, 1)

        if not np.allclose(source_indexes, target_indexes):
            print("docsが正しくソートできていない")

        jptimes = np.vectorize(lambda doc: extractFieldsFromDoc(doc, "JPtime"))(docs)
        dts_per_day = np.vectorize(isoformat2dt)(jptimes)

        qs_per_day = np.array(extractFieldsFromDocs(docs, "solarIrradiance(kw/m^2)"))

        if is_first_loop:
            span_last_dt = (  # 取得する期間列の末尾の日時
                dts_per_day[0]  # dts_per_day[0]から{fixedDaysLen}日後をlast_dtとする
                + datetime.timedelta(days=span_int)
                + datetime.timedelta(hours=span_float * 24)
            )
            is_first_loop = False

        if np.any(dts_per_day > span_last_dt):
            # dts_per_dayの並びの中にlast_dtが存在する
            mask = dts_per_day <= span_last_dt
            dts_under_last_dt_per_day = dts_per_day[mask]
            qs_under_last_dt_per_day = qs_per_day[mask]

            q_all = np.append(q_all, qs_under_last_dt_per_day)
            dt_all = np.append(dt_all, dts_under_last_dt_per_day)

            break

        q_all = np.append(q_all, qs_per_day)
        dt_all = np.append(dt_all, dts_per_day)

        dt_crr_fetching = dt_crr_fetching + datetime.timedelta(days=1)

    return [dt_all, q_all]
