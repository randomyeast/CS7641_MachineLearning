import numpy as np
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from nltk.corpus import stopwords
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC


def remove_stop_words(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return (text)


def construct_sentences(data):
    cleaned_sentences = []
    for index, row in enumerate(data):
        cleaned_sentence = TaggedDocument(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)])
        cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences


def get_data(path,vector_dimension=300):
    data = pd.read_csv(path)

    missing_rows = list(np.where(data['text']!=data['text']))[0]
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)
    sentences = []
    for i in range(len(data)):
        if data['text'][i]:
            sentences.append(remove_stop_words(data['text'][i]))
    x = construct_sentences(sentences)
    y = data['label'].values

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.epochs)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    xtrain,xtest = [],[]
    ytrain = y[:train_size]
    ytest = y[train_size:len(y)]
    for i in range(len(x)):
        if i<train_size:
            xtrain.append(text_model.docvecs['Text_' + str(i)])
        else:
            xtest.append(text_model.docvecs['Text_' + str(i)])
    return np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)





