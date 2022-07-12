from elasticsearch import Elasticsearch
import pandas as pd
import codecs
import datetime

# Elasticsearchクライアント作成
es = Elasticsearch("http://localhost:9200")

filepath = "data/DougoSyou/1809010000.csv"
with codecs.open(filepath, "r", "Shift-JIS", "ignore") as file:
    df = pd.read_csv(file, delimiter=",", skiprows=[1])

    for i, row in df.iterrows():
        year = int(row["TIME"][:4])
        month = int(row["TIME"][5:7])
        day = int(row["TIME"][8:10])
        hour = int(row["TIME"][11:13])
        minute = int(row["TIME"][14:])
        # tzinfoでタイムゾーンJST(UTC+9)を指定しないとKibana上でUTC時刻として登録され、
        # ブラウザ環境からタイムゾーンをAsia/Tokyoとして推定し、
        # 自動的にUTCからJSTに変換されて表示されるので9時間進んで表示される
        # https://discuss.elastic.co/t/kibana-9/158683/2
        row = {
            "JPtime": datetime.datetime(
                year,
                month,
                day,
                hour,
                minute,
                tzinfo=datetime.timezone(datetime.timedelta(hours=9)),
            ),
            "solarIrradiance(kw/m^2)": float(row["日射強度"]),
            "airTemperature(℃)": float(row["外気温度"]),
            "dc-pw(kw)": float(row["直流電力"]),
            "ac-pw(kw)": float(row["交流電力"]),
        }
        es.create(index="solars", id=i + 1, document=row)

# 内部接続を閉じる
es.close()
