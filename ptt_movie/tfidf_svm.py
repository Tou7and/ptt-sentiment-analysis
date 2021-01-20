#!/usr/bin/env python
# coding: utf-8
""" Sentiment Analysis using PTT movie articles """
import logging
import numpy as np
from random import shuffle
import sys
from time import time
import matplotlib.pyplot as plt
from glob import glob
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics

jieba.load_userdict("/Users/mac/projects/ptt-sentiment-analysis/emotion_classifier/data/dict.txt.big")

def preprocess_textfiles(src_txt):
    with open(src_txt, 'r') as reader:
        text_lines = reader.readlines()
    all_words = ""
    for text in text_lines:
        word_list = jieba.cut(text, cut_all=False)
        words = " ".join(word_list)
        all_words = all_words + words
    return all_words

def make_data_and_label(pos_data, neg_data):
    data = []
    label = []
    for textfile in pos_data:
        data.append(preprocess_textfiles(textfile))
        label.append('pos')
        
    for textfile in neg_data:
        data.append(preprocess_textfiles(textfile))
        label.append('neg')
    return data, label

def sentiment_analyse(sklearn_pipeline, text, ispath=False):
    if ispath:
        words = preprocess_textfiles(text)
    else:
        word_list = jieba.cut(text, cut_all=False)
        words = " ".join(word_list)
    pred = sklearn_pipeline.predict([words])
    return pred[0]

model_path = "./model/tfidf_svm-mk1.joblib" 
pos_files = glob("/Users/mac/projects/ptt-comment-spider/egs/movie_posneg/data/positives/*.txt")
neg_files = glob("/Users/mac/projects/ptt-comment-spider/egs/movie_posneg/data/negatives/*.txt")

if __name__ == "__main__":
    print()
    print("Preprocessing raw data ...")
    print("---"*10)
    shuffle(pos_files)
    shuffle(neg_files)

    print("Splitting data to train/validate/test sets ...")
    pos_train = pos_files[0:int(0.8*len(pos_files))]
    pos_validate = pos_files[int(0.8*len(pos_files)):int(0.9*len(pos_files))]
    pos_test = pos_files[int(0.9*len(pos_files)):]
    neg_train = neg_files[0:int(0.8*len(neg_files))]
    neg_validate = neg_files[int(0.8*len(neg_files)):int(0.9*len(neg_files))]
    neg_test = neg_files[int(0.9*len(neg_files)):]
    print("Number of positive data (train/validate/test):", len(pos_train), len(pos_validate), len(pos_test))
    print("Number of negative data (train/validate/test):", len(neg_train), len(neg_validate), len(neg_test))

    data_train, y_train = make_data_and_label(pos_train, neg_train)
    data_validate, y_validate = make_data_and_label(pos_validate, neg_validate)
    data_test, y_test = make_data_and_label(pos_test, neg_test)

    print()
    print("Training Model ...")
    print("---"*10)
    sentiment_pipe = Pipeline([
         ('tfidf', TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=None)),
         ('svm', LinearSVC(penalty="l1", dual=False, tol=1e-3))])
    sentiment_pipe.fit(data_train, y_train)

    pred_validate = sentiment_pipe.predict(data_validate)
    score = metrics.accuracy_score(y_validate, pred_validate)
    print("validation accuracy: %0.4f" % score)
    print(metrics.confusion_matrix(y_validate, pred_validate))
    print()

    print("Saving model ...")
    from joblib import dump, load
    dump(sentiment_pipe, model_path)

    print()
    print("Testing ...")
    print("---"*10)
    new_pipe = load(model_path)

    pred_test = new_pipe.predict(data_test)
    score = metrics.accuracy_score(y_test, pred_test)
    print("testing accuracy: %0.3f" % score)
    print(metrics.confusion_matrix(y_test, pred_test))

