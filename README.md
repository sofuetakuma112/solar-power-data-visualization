# セットアップ
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
