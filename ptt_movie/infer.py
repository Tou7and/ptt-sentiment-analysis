from joblib import dump, load
from tfidf_svm import sentiment_analyse
sentiment_pipe = load("model/tfidf_svm-mk1.joblib")

sample1 = "劇情不錯"
sample2 = "這是一部大爛片。"

print(sample1, ":", sentiment_analyse(sentiment_pipe, sample1))
print(sample2, ":", sentiment_analyse(sentiment_pipe, sample2))
