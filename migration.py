from elasticsearch import Elasticsearch

# Elasticsearchクライアント作成
es = Elasticsearch("http://localhost:9200")

# インデックス一覧の取得
indices = es.cat.indices(index="*", h="index").splitlines()
# 一度すべてのインデックスを削除する
for index in indices:
    es.indices.delete(index=index)

# マッピングを作成
mapping = {
    "mappings": {
        "properties": {
            "JPtime": {"type": "date"},
            "solarIrradiance(kw/m^2)": {"type": "float"},
            "airTemperature(℃)": {"type": "float"},
            "dc-pw(kw)": {"type": "float"},
            "ac-pw(kw)": {"type": "float"},
        }
    }
}
# マッピングを指定してインデックスを作成
es.indices.create(index="solars", body=mapping)

# 内部接続を閉じる
es.close()
