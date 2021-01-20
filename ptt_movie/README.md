# Sentiment Analysis using PTT Movie data
用 PTT 電影版上的好雷/負雷文章來訓練中文情感分類器。

# Steps
- 透過爬蟲抓取 PTT 資料
爬蟲腳本：<https://github.com/Tou7and/ptt-comment-spider/blob/main/egs/movie_posneg/parse.py>

- `tfidf_svm.py`
  - 前處理：將資料分成訓練/驗證/測試集
  - 抽取特徵：透過結巴詞分詞後抽取使用 tf-idf 特徵
  - 訓練/驗證
  - 儲存模型/測試

- `infer.py`
  - 模型調用

