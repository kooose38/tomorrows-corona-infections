# 明日の全国コロナ感染者数の予測  
---  
  
データセット: [厚生労働省オープンデータ](https://www.mhlw.go.jp/stf/covid-19/open-data.html)  

* 新規陽性者数の推移（日別）

* 入院治療等を要する者等推移

* 死亡者数（累積）

* 重症者数の推移  
  
---  
  
使い方:  
1. コマンドを実行する  
```
$ python3 predict.py  
sucessfully download csv files  
明日の感染者数は6649.6人と予想されます  
```  



---  
  
注釈: 時期により大幅な分布傾向が異なるので、直近３か月に絞ったデータでモデルを学習した。よって、時間経過に伴ってモデルの再現性が低下することを示します。  
モデルの訓練コードは`./notebook`に配置。  
  
---  
  

追記: テーブルデータでの予測モデルを使用した結果  
  
![alt](./img/covid19.png)

