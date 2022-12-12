import sys
import os

sys.path.append(os.path.abspath(".."))

from elasticsearch import Elasticsearch
import pickle
import datetime

from utils.file import get_pickle_file_path_by_datetime

from dotenv import load_dotenv

load_dotenv(f"{os.getcwd()}/.env")


def fetchDocsByDatetime(dt):
    es = Elasticsearch(
        "http://133.71.201.197:9200",
        http_auth=(
            os.getenv("RECYCLE_ELASTIC_USER_NAME"),
            os.getenv("RECYCLE_ELASTIC_PASSWORD"),
        ),
    )

    indexName = "pcs_recyclekan"

    data_type = "raw"
    if not os.path.exists(f"{os.getcwd()}/pickles/{data_type}"):
        os.mkdir(f"{os.getcwd()}/pickles/{data_type}")

    filePath = get_pickle_file_path_by_datetime(dt, data_type)
    # すでにPickleファイルが存在するならElasticSearchから取得しない
    if os.path.isfile(filePath):
        return

    dt_next = dt + datetime.timedelta(days=1)
    query = {
        "query": {
            "range": {
                "JPtime": {
                    "gte": f"{dt.year}-{str(dt.month).zfill(2)}-{str(dt.day).zfill(2)}T00:00:00",
                    "lt": f"{dt_next.year}-{str(dt_next.month).zfill(2)}-{str(dt_next.day).zfill(2)}T00:00:00",
                },  # JST時間をUTC時間として登録しているのでUTC時間として検索する必要がある
            }
        }
    }

    s_time = "2m"
    data = es.search(
        index=indexName,
        scroll=s_time,
        body=query,
        size=1000,
        request_timeout=150,
    )

    s_id = data["_scroll_id"]
    s_size = data["hits"]["total"]["value"]  # 残りの検索対象の件数??
    result = data["hits"]["hits"]
    while s_size > 0:
        data = es.scroll(
            scroll_id=s_id, scroll=s_time, request_timeout=150
        )  # scroll: スクロール時の検索コンテキストを保持するための期間
        s_id = data["_scroll_id"]
        s_size = len(data["hits"]["hits"])
        result.extend(data["hits"]["hits"])

    with open(filePath, "wb") as f:
        pickle.dump(result, f)

    es.close()  # 内部接続を閉じる


# 2日以上のデータをelasticsearchから取得する関数
def fetchDocsByPeriod(fromDt, toDt):
    # 1日おきにfetchDocsByDatetimeを呼び出してpickleファイルを書き込んでいく
    # datetime同士の減算はtimedeltaのインスタンスになる
    dtDiff = toDt - fromDt
    dt_crr = fromDt
    dt_now = datetime.datetime.now()
    for _ in range(dtDiff.days + 1):
        if dt_crr > dt_now:
            break
        fetchDocsByDatetime(dt_crr)
        dt_crr = dt_crr + datetime.timedelta(days=1)


if __name__ == "__main__":
    fetchDocsByPeriod(datetime.datetime(2022, 1, 1), datetime.datetime(2022, 10, 20))
