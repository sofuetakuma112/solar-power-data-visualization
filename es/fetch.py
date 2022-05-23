from elasticsearch import Elasticsearch
import pickle
import datetime
import os


def fetchDocsByDatetime(dt_crr, filePath):
    # Elasticsearchインスタンスを作成
    es = Elasticsearch("http://133.71.201.197:9200", http_auth=("takenaka", "takenaka"))

    indexName = "pcs_recyclekan"

    # すでにPickleファイルが存在するならElasticSearchから取得しない
    if os.path.isfile(filePath):
        return

    dt_prev = dt_crr + datetime.timedelta(days=-1)
    query = {
        "query": {
            "range": {
                # TODO: utctimeではなく、JPTimeのrangeで検索したほうがわかりやすい？
                "utctime": {  # utctimeはkibana上で自動的にJSTに変換して表示するのを前提に、計測したタイミングのUTC時刻 - 9hをUTC時刻として登録しているので
                    "gte": f"{dt_prev.year}-{str(dt_prev.month).zfill(2)}-{str(dt_prev.day).zfill(2)}T15:00:00",
                    "lt": f"{dt_crr.year}-{str(dt_crr.month).zfill(2)}-{str(dt_crr.day).zfill(2)}T15:00:00",
                },  # JST時間をUTC時間として登録しているのでUTC時間として検索する必要がある
            }
        }
    }

    num = 10
    s_time = "2m"
    data = es.search(
        index=indexName,
        scroll=s_time,
        body=query,
        size=num,
        request_timeout=150,
    )

    s_id = data["_scroll_id"]
    s_size = data["hits"]["total"]["value"]
    result = data["hits"]["hits"]
    while s_size > 0:
        data = es.scroll(scroll_id=s_id, scroll=s_time, request_timeout=150)
        s_id = data["_scroll_id"]
        s_size = len(data["hits"]["hits"])
        result.extend(data["hits"]["hits"])

    # for document in result:
    #     print(document["_source"]["utctime"])

    with open(filePath, "wb") as f:
        pickle.dump(result, f)

    # 内部接続を閉じる
    es.close()
