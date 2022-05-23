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
            "name": {"type": "text"},
            "age": {"type": "long"},
            "email": {"type": "text"},
        }
    }
}
# マッピングを指定してインデックスを作成
es.indices.create(index="students", body=mapping)

# print(es.indices.get_mapping(index="students"))

# 登録したいドキュメント
student1 = {
    "name": "Taro",
    "age": 36,
}
es.create(index='students', id=1, document=student1)
student2 = {
    "name": "Jiro",
    "age": 32,
    "email": "jiro@example.com"
}
es.create(index='students', id=2, document=student2)

# print(es.get_source(index="students", id=1))

mapping = {
    "properties": {
        "student_number": {"type": "long"}
    }
}
# インデックスのマッピングを更新
es.indices.put_mapping(index="students", body=mapping)
# print(es.indices.get_mapping(index="students"))

student3 = {
    "name": "Saburo",
    "age": 29,
    "email": "saburo@example.com",
    "student_number": 1234
}
es.create(index='students', id=3, document=student3)

# print(es.get_source(index="students", id=1))

# 内部接続を閉じる
es.close()
