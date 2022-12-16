# セットアップ
## pythonライブラリ
- numpy
- pandas
- matplotlib
- japanize-matplotlib
- pyqt5
- python-dotenv
- elasticsearch

## コンテナ立ち上げ
`docker compose up`

## Enrollment tokenの発行
ElasticsearchをインストールしているDockerコンテナに入る
```sh
docker exec -it elasticsearch-tutorial-elasticsearch-1 /bin/sh
cd /usr/share/elasticsearch
bin/elasticsearch-create-enrollment-token --scope kibana
```

## Kibanaから認証コードを発行する
KibanaをインストールしているDockerコンテナに入る
```sh
docker exec -it elasticsearch-tutorial-kibana-1 /bin/sh
cd /usr/share/kibana
bin/kibana-verification-code
```

# ElasticsearchのインデックスをKibana上でモニタリングする
1. http://localhost:5601/app/management/kibana/dataViews にアクセスしてDataViewを作成する
2. http://localhost:5601/app/discover から見れるようになる
